package com.idibon.ml.train.alloy

import java.util.Random

import com.idibon.ml.alloy.{BaseAlloy, HasTrainingSummary, Alloy}
import com.idibon.ml.common.Engine
import com.idibon.ml.feature.{Builder, Buildable}
import com.idibon.ml.predict.ml.TrainingSummary
import com.idibon.ml.predict.ml.metrics._
import com.idibon.ml.predict.{Span, Label, PredictResult}
import com.idibon.ml.train.TrainOptions
import com.idibon.ml.train.alloy.evaluation.{Granularity, AlloyEvaluator}
import com.typesafe.scalalogging.StrictLogging
import org.json4s.JsonAST.JObject

/**
  * This creates learning curves, by using the cross validation alloy forge.
  *
  * For each portion, it kicks off a cross validation forge which returns an
  * alloy of cross validation metrics for that portion. We then extract those
  * metrics to create some data points. For posterity all the portion cross
  * validation metrics are also included in the alloy.
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>" on 3/21/16.
  * @param engine the engine
  * @param name the name to give the output alloy & and anything else created will be based off this name.
  * @param labels the labels for the current task
  * @param forge the forge to use for all cross validation
  * @param numFolds the number of folds for each cross validation run
  * @param portions the training data size portions to use
  * @param foldSeed the random seed used to help split up the data
  * @tparam T the parameter whether we're doing classification or spans
  */
class LearningCurveAlloyForge[T <: PredictResult with Buildable[T, Builder[T]]](
  engine: Engine,
  override val name: String,
  override val labels: Seq[Label],
  forge: AlloyForge[T],
  numFolds: Int,
  portions: Array[Double],
  foldSeed: Long)
  extends AlloyForge[T] with HasAlloyForges[T] with KFoldDataSetCreator with StrictLogging {

  override val forges: Seq[AlloyForge[T]] = portions.map(portion => {
    new CrossValidatingAlloyForge[T](
      engine, s"$name-$portion", labels, forge, numFolds, portion, foldSeed, true)
  })

  /**
    * Synchronous method that creates the alloy.
    *
    * @param options
    * @param evaluator
    * @return
    */
  override def doForge(options: TrainOptions, evaluator: AlloyEvaluator): Alloy[T] = {

    val alloys = this.forgeForges(options, evaluator)
      .map(x => x.asInstanceOf[BaseAlloy[T] with HasTrainingSummary])
    val portionAlloys = alloys.map(x => (x.name.split("-").reverse.head.toDouble, x))
    // get portion summaries
    val portionSummaries = getXValPortionSummaries(portionAlloys)
    // so now we have results for each portion -- already averaged, create LC metrics
    val metricTypeToPortions = transformAndFilterToWantedMetrics(portionSummaries)
    val lcMetrics = createLearningCurveMetrics(metricTypeToPortions)
    val lcSummary = new TrainingSummary(s"$name-LC", lcMetrics)
    logger.info(lcSummary.toString)
    new BaseAlloy[T](s"$name-LC", labels, Map()) with HasTrainingSummary {
      override def getTrainingSummaries = {
        Some(Seq(lcSummary) ++ portionSummaries.map(_._2))
      }
    }
  }

  /**
    * Returns the appropriate evaluator for this alloy.
    *
    * @param engine
    * @param taskType
    * @return
    */
  override def getEvaluator(engine: Engine, taskType: String): AlloyEvaluator = {
    forge.getEvaluator(engine, taskType)
  }

  /**
    * Gets the training summaries that come from cross validation alloys.
    *
    * @param portionAlloys
    * @return
    */
  def getXValPortionSummaries(portionAlloys: Seq[(Double, BaseAlloy[T] with HasTrainingSummary)]):
  Seq[(Double, TrainingSummary)] = {
    portionAlloys.map({ case (portion, alloy) =>
      (portion, alloy.getTrainingSummaries)
    }).collect({ case (portion: Double, Some(summary)) => (portion, summary) })
      .flatMap({ case (portion, summary) => summary.map(ts => (portion, ts)) })
      // this probably isn't needed, but it's a safeguard anyway
      .filter({ case (portion, ts) => ts.identifier.endsWith(CrossValidatingAlloyForge.SUFFIX) })
      // filter to document or token level summaries only
      .filter({case (portion, ts) =>
      ts.getNotesValues(AlloyEvaluator.GRANULARITY)
        .exists(s =>
          s.equals(Granularity.Document.toString) || s.equals(Granularity.Token.toString))
    })
  }

  /**
    * Creates points for plotting learning curve metrics.
    *
    * This creates LearningCurveLabelF1, LearningCurveLabelPrecision, LearningCurveLabelRecall
    * and LearningCurveF1 metrics.
    *
    * @param metrics
    * @return
    */
  def createLearningCurveMetrics(metrics: Map[MetricTypes, Seq[(Double, Metric with Buildable[_, _])]]):
  Seq[Metric with Buildable[_, _]] = {
    metrics.flatMap({case (metricType, portionMetrics) =>
      val newMetric: Seq[Metric with Buildable[_, _]] = metricType match {
        case MetricTypes.LabelF1 => {
          createLabelPointsMetrics(MetricTypes.LearningCurveLabelF1, MetricClass.Alloy, portionMetrics)
        }
        case MetricTypes.LabelPrecision => {
          createLabelPointsMetrics(MetricTypes.LearningCurveLabelPrecision, MetricClass.Alloy, portionMetrics)
        }
        case MetricTypes.LabelRecall => {
          createLabelPointsMetrics(MetricTypes.LearningCurveLabelRecall, MetricClass.Alloy, portionMetrics)
        }
        case MetricTypes.F1 => {
          //sort into each label
          val points = portionMetrics.map({case (portion, metric) =>
            (portion.toFloat, metric match {case m: FloatMetric => m.float})
          })
          Seq(new PointsMetric(MetricTypes.LearningCurveF1, MetricClass.Alloy, points))
        }
        case other => throw new IllegalStateException(s"Error; should not have encountered ${other}.")
      }
      newMetric
    }).toSeq
  }

  /**
    * This creates a label ploints metrics object.
    *
    * This represents points that can be plotted for a label.
    *
    * @param metricType
    * @param metricClass
    * @param portionMetrics
    * @return
    */
  def createLabelPointsMetrics(metricType: MetricTypes,
                               metricClass: MetricClass.Value,
                               portionMetrics: Seq[(Double, Metric with Buildable[_, _])]) = {
    //sort into each label
    val byLabel = portionMetrics.groupBy({case (portion, metric:LabelFloatMetric) => metric.label})
    byLabel.map({case (label, labelPortionMetrics) =>
      val points = labelPortionMetrics.map({case (portion, metric) =>
        (portion.toFloat, metric match {case m: LabelFloatMetric => m.float})
      })
      new LabelPointsMetric(metricType, metricClass, label, points)
    }).toSeq
  }

  /**
    * Helper method to transform a sequence of (portion, training summary) into just a sequence
    * of (portion, metric).
    *
    * i.e. this flattens the training summaries into just metrics by portion.
    *
    * @param portionSummaries
    * @return
    */
  def transformAndFilterToWantedMetrics(portionSummaries: Seq[(Double, TrainingSummary)]):
  Map[MetricTypes, Seq[(Double, Metric with Buildable[_, _])]] = {
    portionSummaries
      // get all metrics out
      .flatMap({ case (portion, ts) =>
      ts.metrics.map(m => (portion, m))
    })
      // make sure they're the ones we want
      .filter({case (portion, metric) => metric.metricClass == MetricClass.Alloy})

      .groupBy({ case (portion, metric) => metric.metricType })
      .filter({ case (metricType, _) =>
        (metricType == MetricTypes.LabelF1) ||
          (metricType == MetricTypes.LabelPrecision) ||
          (metricType == MetricTypes.LabelRecall) ||
          (metricType == MetricTypes.F1) // gratuitous since it's not used externally...
      })
  }
}

object LearningCurveSpanAlloyForge extends ((Engine, String, Seq[Label], JObject) => AlloyForge[_]) {

  override def apply(engine: Engine, name: String, labels: Seq[Label], json: JObject):
  LearningCurveAlloyForge[Span] = {
    implicit val formats = org.json4s.DefaultFormats
    val config = json.extract[LearningCurveAlloyForgeConfig]
    //TODO: check version
    val forge = AlloyForge.apply[Span](engine, config.forgeName, name, labels, config.forgeConfig)
    val numFolds = config.numFolds
    val portions = config.portions
    val foldSeed = config.foldSeed
    new LearningCurveAlloyForge[Span](engine, name, labels, forge, numFolds, portions, foldSeed)
  }
}

// === JSON configuration schema ===
case class LearningCurveAlloyForgeConfig(version: String,
                                         forgeName: String,
                                         forgeConfig: JObject,
                                         numFolds: Int = 5,
                                         portions: Array[Double] = Array[Double](0.25, 0.5, 0.625, 0.75, 0.8125, 0.875, 0.9375, 1.0),
                                         foldSeed: Long = new Random().nextLong())
