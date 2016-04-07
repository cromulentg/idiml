package com.idibon.ml.train.alloy

import java.util.Random

import com.idibon.ml.alloy.{HasTrainingSummary, BaseAlloy, Alloy}
import com.idibon.ml.common.Engine
import com.idibon.ml.feature.{Builder, Buildable}
import com.idibon.ml.predict.ml.TrainingSummary
import com.idibon.ml.predict.ml.metrics.{MetricClass, FloatMetric, MetricTypes}
import com.idibon.ml.predict.{Span, Label, PredictResult}
import com.idibon.ml.train.TrainOptions
import com.idibon.ml.train.alloy.evaluation.{Granularity, AlloyEvaluator}
import com.typesafe.scalalogging.StrictLogging
import org.json4s.JsonAST.JObject

/**
  * This forge pits different alloy forges head to head and chooses
  * the best one; returning an alloy trained on all the data that got
  * the best results from the cross validation forge run.
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>" on 3/21/16.
  * @param engine the engine with the spark context.
  * @param name the base name for the alloys this will create.
  * @param labels the labels that are represented in the data.
  * @param baseForges the forges to compete against each other.
  * @param numFolds the number of folds to use for cross validation.
  * @param foldSeed the fold seed to use to split up the data.
  * @tparam T the kind of alloy to make.
  */
class CompetitiveAlloyForge[T <: PredictResult with Buildable[T, Builder[T]]](
   engine: Engine,
   override val name: String,
   override val labels: Seq[Label],
   val baseForges: Map[String, AlloyForge[T]],
   numFolds: Int,
   foldSeed: Long)
  extends AlloyForge[T] with HasAlloyForges[T] with KFoldDataSetCreator with StrictLogging {

  /** Creates the forges that we'll be comparing against **/
  override val forges: Seq[AlloyForge[T]] = baseForges.map({case (trainerName, forge) =>
    new CrossValidatingAlloyForge[T](
      engine, s"$name-$trainerName", labels, forge, numFolds, 1.0, foldSeed)
  }).toSeq

  /**
    * Synchronous method that creates the alloy.
    *
    * It delegates to training all the alloys, and then finds the best one
    * according to max F1.
    *
    * @param options
    * @param evaluator
    * @return
    */
  override def doForge(options: TrainOptions, evaluator: AlloyEvaluator): Alloy[T] = {
    val alloys = this.forgeForges(options, evaluator)
      .map(x => x.asInstanceOf[BaseAlloy[T] with HasTrainingSummary])
    val maxAlloy = CompetitiveAlloyForge.findMaxF1[T](alloys)
    val maxAlloyName = maxAlloy.name
    logger.info(s"Max F1 alloy was $maxAlloyName")
    /* create new alloy with extra training summaries from other alloys
       removing their full training run metrics */
    val allSummaries = alloys.map(a => a.getTrainingSummaries)
      .collect({ case Some(s) => s})
      .flatten
      //TODO: might have to do a bit more here...
      .filter(ts => ts.identifier.contains(maxAlloyName) ||
        ts.identifier.endsWith(CrossValidatingAlloyForge.SUFFIX))
    allSummaries.foreach(summary => logger.info(summary.toString))
    new BaseAlloy[T](maxAlloyName, labels, maxAlloy.models) with HasTrainingSummary {
      override def getTrainingSummaries = {
        Some(allSummaries)
      }
    }
  }

  /**
    * Returns the appropriate evaluator for this alloy.
    *
    * Assumes that we can just return the first forge's evaluator and it'll work.
    * This should because each evaluator is only tied to a particular task type,
    * which is the same for all forges.
    *
    * @param engine
    * @param taskType
    * @return
    */
  override def getEvaluator(engine: Engine, taskType: String): AlloyEvaluator = {
    forges.head.getEvaluator(engine, taskType)
  }
}

/**
  * Companion object to house the static methods.
  */
object CompetitiveAlloyForge {

  /**
    * Helper function find the alloy forge and training summary with the best F1
    * score.
    *
    * TODO: extend this so that we can configure whether we want max F1, or an average
    * of F1 across all labels, etc...
    *
    * @param averagedResults
    * @return
    */
  def findMaxF1[T <: PredictResult with Buildable[T, Builder[T]]](
   averagedResults: Seq[BaseAlloy[T] with HasTrainingSummary]): BaseAlloy[T] with HasTrainingSummary = {
    //find max by finding the value to use
    val maxF1 = averagedResults.maxBy({case alloy =>
      alloy.getTrainingSummaries.map(summaries => {
        val candidateSummaries = filterToAppropriateSummaries(summaries)
        candidateSummaries.flatMap(_.metrics)
          .find(m => m.metricType == MetricTypes.F1 || m.metricType == MetricTypes.MacroF1)
          .map(_.asInstanceOf[FloatMetric].float)
          .getOrElse(0.0f)
      })
    })
    maxF1
  }

  /**
    * Method to filter to the appropriate summaries.
    *
    * @param summaries
    * @return
    */
  def filterToAppropriateSummaries(summaries: Seq[TrainingSummary]): Seq[TrainingSummary] = {
    summaries.filter(ts => {
      ts.identifier.endsWith(CrossValidatingAlloyTrainer.SUFFIX)
    }).filter(ts => {
      // filter by granularity -- so we're only getting the max of the ones we want.
      val notes = ts.getNotesValues(AlloyEvaluator.GRANULARITY)
      // if there are no granularity notes, assume this training summary is fine.
      if (notes.nonEmpty) {
        notes.exists(v => {
          v == Granularity.Document.toString || v == Granularity.Token.toString
        })
      } else {
        true
      }
    })
  }
}

/**
  * Companion object that creates competitive span alloy forges.
  */
object CompetitiveSpanAlloyForge extends ((Engine, String, Seq[Label], JObject) => AlloyForge[_]) {
  /**
    * This method is registered in the AlloyForge registery and used to instantiate a forge from
    * the right configuration.
    *
    * @param engine the engine with context.
    * @param name the name to be used in naming the output.
    * @param labels the labels used.
    * @param json the configuration to inspect.
    * @return CompetitiveAlloyForge[Span]
    */
  override def apply(engine: Engine, name: String, labels: Seq[Label], json: JObject):
  CompetitiveAlloyForge[Span] = {
    implicit val formats = org.json4s.DefaultFormats
    val config = json.extract[CompetitiveAlloyForgeConfig]
    // TODO: Should probably check the version
    val forges = config.forges.map({case (givenName, forgeConfig) =>
      (givenName, AlloyForge.apply[Span](
        engine, forgeConfig.forgeName, givenName, labels, forgeConfig.forgeConfig))
    })
    new CompetitiveAlloyForge[Span](engine, name, labels, forges, config.numFolds, config.foldSeed)
  }
}

// -- JSON config classes
case class CompetitiveAlloyForgeConfig(forges: Map[String, CompetingForge],
                                       numFolds: Int = 5,
                                       foldSeed: Long = new Random().nextLong(),
                                       version: String)

case class CompetingForge(forgeName: String, forgeConfig: JObject, version: String)
