package com.idibon.ml.train.alloy

import com.idibon.ml.alloy.{BaseAlloy, HasTrainingSummary, Alloy}
import com.idibon.ml.common.Engine
import com.idibon.ml.predict.Classification
import com.idibon.ml.predict.ml.metrics.{FloatMetric, MetricTypes}
import com.typesafe.scalalogging.StrictLogging
import org.json4s.JsonAST.JObject

/**
  * This trainer pits different alloy trainers head to head and chooses
  * the best one; returning an alloy trained on all the data that got
  * the best results from the cross validation run.
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>" on 3/2/16.
  * @param builder
  */
@deprecated("CompetitiveAlloyForge is the new way.", "Since 2015-04-03")
class CompetitiveAlloyTrainer(builder: CompetitiveAlloyTrainerBuilder)
  extends AlloyTrainer with StrictLogging {
  val engine: Engine = builder.engine
  val trainers: Map[String, AlloyTrainer] = builder.trainerBuilders
    .map({ case (name, a) => (name, a.build(builder.engine))})
  val numFolds: Int = builder.numFolds
  val portion: Double = 1.0 // should always be 1.0 no need to make it configurable.
  val foldSeed: Long = builder.foldSeed


  /** Trains a model and generates an Alloy from it
    *
    * Callers must provide a callback function which returns a traversable
    * list of documents; this function will be called multiple times, and
    * each invocation of the function must return an instance that will
    * traverse over the exact set of documents traversed by previous instances.
    *
    * Traversed documents should match the format generated by
    * idibin.git:/idibin/bin/open_source_integration/export_training_to_idiml.rb
    *
    * { "content": "Who drives a chevy maliby Would you recommend it?
    * "metadata": { "iso_639_1": "en" },
    * "annotations: [{ "label": { "name": "Intent" }, "isPositive": true }]}
    *
    * @param name          - a user-friendly name for the Alloy
    * @param docs          - a callback function returning a traversable sequence
    *                      of JSON training documents, such as those generated by export_training_to_idiml.rb
    * @param labelsAndRules a callback function returning a traversable sequence
    *                      of JSON Config. Should only be one line,   generated by export_training_to_idiml.rb.
    * @param config        training configuration parameters. Optional.
    * @return an Alloy with the trained model
    */
  override def trainAlloy(name: String,
                          docs: () => TraversableOnce[JObject],
                          labelsAndRules: JObject,
                          config: Option[JObject]): Alloy[Classification] = {
    implicit val formats = org.json4s.DefaultFormats
    val uuidToLabel = (labelsAndRules \ "uuid_to_label").extract[JObject]
    val labels = uuidToLabelGenerator(uuidToLabel)

    // for each trainer
    val alloys = trainers.map({case (trainerName, trainer) =>
      // delegate to xval trainer
      val cvat = new CrossValidatingAlloyTrainer(engine, trainer, numFolds, portion, foldSeed)
      cvat.trainAlloy(s"$name-$trainerName", docs, labelsAndRules, config)
        .asInstanceOf[BaseAlloy[Classification] with HasTrainingSummary]
    }).toList.toSeq
    // find max from results
    val maxAlloy = findMaxF1(alloys)
    val maxAlloyName = maxAlloy.name
    logger.info(s"Max F1 alloy was $maxAlloyName")
    /* create new alloy with extra training summaries from other alloys
       removing their full training run metrics */
    val allSummaries = alloys.map(a => a.getTrainingSummaries)
      .collect({ case Some(s) => s})
      .flatten
      .filter(ts => ts.identifier.contains(maxAlloyName) ||
        ts.identifier.endsWith(CrossValidatingAlloyTrainer.SUFFIX))
    allSummaries.foreach(summary => logger.info(summary.toString))
    new BaseAlloy[Classification](maxAlloyName, labels, maxAlloy.models) with HasTrainingSummary {
      override def getTrainingSummaries = {
        Some(allSummaries)
      }
    }
  }


  /**
    * Helper function find the alloy trainer and training summary with the best F1
    * score.
    *
    * TODO: extend this so that we can configure whether we want max F1, or an average
    * of F1 across all labels, etc...
    *
    * @param averagedResults
    * @return
    */
  def findMaxF1(averagedResults:  Seq[BaseAlloy[Classification] with HasTrainingSummary]):
  BaseAlloy[Classification] with HasTrainingSummary = {
    // we don't care about portion for AlloyXValidation since it should just be 1.0
    val maxF1 = averagedResults.maxBy({case alloy =>
      alloy.getTrainingSummaries match {
        case Some(summaries) => {
          // find the summary from cross validation & then find the F1 metric in it
          summaries.filter(ts =>
            ts.identifier.endsWith(CrossValidatingAlloyTrainer.SUFFIX)
          ).head
            .metrics
            .find(m => m.metricType == MetricTypes.F1)
            .get.asInstanceOf[FloatMetric].float
        }
        case None => 0.0f
      }
    })
    maxF1
  }
}
