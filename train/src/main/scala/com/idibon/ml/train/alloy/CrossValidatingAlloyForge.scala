package com.idibon.ml.train.alloy

import java.util.Random

import com.idibon.ml.alloy.{HasTrainingSummary, BaseAlloy, Alloy}
import com.idibon.ml.common.Engine
import com.idibon.ml.feature.{SequenceGenerator, Builder, Buildable}
import com.idibon.ml.predict.{Span, Label, PredictResult, Classification}
import com.idibon.ml.predict.ml.TrainingSummary
import com.idibon.ml.predict.ml.metrics._
import com.idibon.ml.train.TrainOptions
import com.idibon.ml.train.alloy.evaluation.AlloyEvaluator
import com.typesafe.scalalogging.StrictLogging
import org.json4s.JsonAST.JObject
import org.json4s._

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Promise, ExecutionContext, Future}
import scala.util.{Try}

/**
  * Cross validating alloy forge.
  *
  * Given, a data set, a portion and a trainer, performs alloy level
  * cross validation and then forges an alloy over all the data.
  *
  * This facilitates getting an idea how a trained alloy will perform
  * on unseen data.
  *
  * The returned alloy contains averaged training summaries from the cross
  * validation, as well as the training metrics from fitting over all the
  * data.
  *
  * Note: the training metrics should be better than the averaged
  * ones, since we're evaluating using the training data, while the
  * averaged metrics were based on the average of the fold hold out sets.
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>" on 3/2/16.
  * @param engine
  * @param name
  * @param labels
  * @param forge
  * @param numFolds
  * @param portion
  * @param foldSeed
  * @param skipFinalTraining
  * @tparam T
  */
class CrossValidatingAlloyForge[T <: PredictResult with Buildable[T, Builder[T]]](
      engine: Engine,
      override val name: String,
      override val labels: Seq[Label],
      forge: AlloyForge[T],
      numFolds: Int,
      portion: Double,
      foldSeed: Long,
      skipFinalTraining: Boolean = false)
  extends AlloyForge[T] with HasAlloyForges[T] with KFoldDataSetCreator with StrictLogging {

  override val forges: Seq[AlloyForge[T]] = Seq(forge)


  /** Initiate a batch training process across all models to produce the Alloy
    *
    * @param options configuration options
    * @param evaluator the evaluator used to measure the underlying alloy.
    * @return an alloy -- with the stats inside as training summaries.
    */
  override def doForge(options: TrainOptions, evaluator: AlloyEvaluator): Alloy[T] =  {
    import ExecutionContext.Implicits.global
    val labelToDouble = labels.zipWithIndex.map({ case (label, index) => (label, index.toDouble) }).toMap
    // create folds -- i.e. data sets
    val folds = createFoldDataSets(options.dataSet.train, numFolds, portion, foldSeed, labelToDouble)
    // get training summaries from alloy
    val summaries = crossValidate(name, evaluator, folds, options)
    // average the results
    val resultsAverage = averageMetrics(s"$name${CrossValidatingAlloyForge.SUFFIX}", summaries)
    logger.info(s"Xval results for - $name:\n ${resultsAverage.toString}")
    if (skipFinalTraining) {
      // create new alloy with just xval summaries -- no model!
      new BaseAlloy[T](name, labels, Map()) with HasTrainingSummary {
        override def getTrainingSummaries = {
          Some(resultsAverage)
        }
      }
    } else {
      // train on all data
      val finalAlloy = forge.forge(options, evaluator)
        .map(a => a.asInstanceOf[BaseAlloy[T] with HasTrainingSummary])
        .map(a => {
          // get training summary
          val finalAlloyTrainingSummary = HasTrainingSummary.getSummaries[T](a)
            .map(ts => new TrainingSummary(s"$name-${ts.identifier}", ts.metrics))
          // create new alloy with combined summaries
          new BaseAlloy[T](name, labels, a.models) with HasTrainingSummary {
            override def getTrainingSummaries = {
              Some(resultsAverage ++ finalAlloyTrainingSummary)
            }
          }
        })
      Await.result(finalAlloy, Duration.Inf)
    }
  }

  /**
    * Method that goes over the folds and delegates training & evaluation.
    *
    * @param name
    * @param evaluator
    * @param folds
    * @return
    */
  def crossValidate(name: String,
                    evaluator: AlloyEvaluator,
                    folds: Seq[TrainingDataSet],
                    baseOptions: TrainOptions): Seq[TrainingSummary] = {
    folds.map(ds => {
      logger.info(s"Beginning fold run: $name-${ds.info.fold}-${ds.info.portion}")
      val alloy = this.forgeForges(
        new TrainOptions(baseOptions.maxTrainTime, ds, baseOptions.labels, baseOptions.rules),
        evaluator).head
      alloy.asInstanceOf[BaseAlloy[T] with HasTrainingSummary].getTrainingSummaries
    }).collect({case Some(ts) => ts}).flatten
  }

  /**
    * Function to average metrics based on metric type.
    *
    * @param summaryName the name to give the new training summary.
    * @param trainingSummaries the training summaries to average metrics for.
    * @return a single training summary with averaged metrics.
    */
  def averageMetrics(summaryName: String,
                     trainingSummaries: Seq[TrainingSummary]): Seq[TrainingSummary] = {
    // split by notes
    val byGranularity = trainingSummaries.groupBy(ts => getGranularity(ts)) // get the granularity out
    byGranularity.map({case (_, summaries) =>
      // average the training summaries
      val averagedSummmary = TrainingSummary.averageSummaries(summaryName, summaries, MetricClass.Alloy)
      // create additional metrics
      val extraMetrics = createAdditionalMetrics(averagedSummmary.metrics)
      // filter unwanted metrics
      val filteredMetrics = filterUnwantedMetrics(averagedSummmary.metrics)
      // create summary
      new TrainingSummary(summaryName, filteredMetrics ++ extraMetrics)
    }).toSeq
  }

  /**
    * Filters unwanted metrics that cross validation wants to remove.
    *
    * Specifically it removes MetricTypes.LabelProbabilities metrics.
    *
    * @param metrics metrics to filter
    * @return a filtered list of metrics
    */
  def filterUnwantedMetrics(metrics: Seq[Metric with Buildable[_, _]]) = {
    metrics.filterNot(metric =>
      metric.isInstanceOf[LabelFloatListMetric] &&
        metric.metricType == MetricTypes.LabelProbabilities)
  }

  /**
    * Helper method to inspect a training summary and get the granularity out.
    *
    * @param ts the training summary to inspect
    * @return a string value representing the granularity
    */
  private def getGranularity(ts: TrainingSummary): String = {
    ts.metrics
      .filter(m => m.metricType == MetricTypes.Notes)
      .collect({ case p: PropertyMetric => p })
      .find(p => p.properties
        .filter({ case (key, value) => key.equals(AlloyEvaluator.GRANULARITY) }).nonEmpty)
      .map(p => p.properties
        .filter({ case (key, value) => key.equals(AlloyEvaluator.GRANULARITY) }).head._2)
      .getOrElse("n/a")
  }

  /**
    * Creates additional metrics that cross validation wants to add to the training summary.
    *
    * Namely, adds:
    *  - max probability
    *  - min probabilty
    *  - computed deciles
    *
    * @param metrics the metrics to use as a base
    * @return
    */
  def createAdditionalMetrics(metrics: Seq[Metric with Buildable[_, _]]) = {
    metrics.filter(metric => metric.metricType ==  MetricTypes.LabelProbabilities)
      .collect({case (m: LabelFloatListMetric) => m})
      .flatMap(m => {
        Seq[Metric with Buildable[_, _]](
          computeProbabilityDeciles(m),
          LabelFloatMetric.computeMinProbability(m),
          LabelFloatMetric.computeMaxProbability(m))
      })
  }

  /**
    * Computes confidence deciles based on a passed in label float list metric.
    *
    * @param metric the metric to compute deciles from.
    * @return a LabelPointsMetric representing LabelConfidenceDeciles
    */
  def computeProbabilityDeciles(metric: LabelFloatListMetric): LabelPointsMetric  = {
    val numConfidences = metric.points.size
    val cut_dq = metric.points.size / CrossValidatingAlloyForge.NUM_QUANTILES
    val cut_dq_rem = metric.points.size % CrossValidatingAlloyForge.NUM_QUANTILES

    var x = Math.max(cut_dq - 1, 0)
    var err = cut_dq_rem

    val quantiles = (1 until CrossValidatingAlloyTrainer.NUM_QUANTILES).map(i => {
      val cut_point = metric.points(x)
      x += cut_dq
      err += cut_dq_rem
      if (err >= CrossValidatingAlloyTrainer.NUM_QUANTILES) {
        x += 1
        err -= CrossValidatingAlloyTrainer.NUM_QUANTILES
      }
      (i.toFloat, cut_point)
    })
    new LabelPointsMetric(
      MetricTypes.LabelConfidenceDeciles, metric.metricClass, metric.label, quantiles.sortBy(x => x._1))
  }

  /**
    * Gets the evaluator for this alloy.
    *
    * @param engine
    * @param taskType
    * @return
    */
  override def getEvaluator(engine: Engine, taskType: String): AlloyEvaluator = {
    forge.getEvaluator(engine, taskType)
  }
}

object CrossValidatingAlloyForge  {
  val NUM_QUANTILES = 10
  val SUFFIX = "-AVG"
  val CURRENT_VERSION = "0.0.1"
}

object CrossValidatingSpanAlloyForge extends ((Engine, String, Seq[Label], JObject) => AlloyForge[_]) {

  override def apply(engine: Engine, name: String, labels: Seq[Label], json: JObject):
  CrossValidatingAlloyForge[Span] = {
    implicit val formats = org.json4s.DefaultFormats
    val config = json.extract[CrossValidatingAlloyForgeConfig]
    //TODO: check version
    val forge = AlloyForge[Span](engine, config.forgeName, name, labels, config.forgeConfig)
    val numFolds = config.numFolds
    val portion = config.portion
    val foldSeed = config.foldSeed
    val skipFinalTraining = config.skipFinalTraining
    new CrossValidatingAlloyForge[Span](
      engine, name, labels, forge, numFolds, portion, foldSeed, skipFinalTraining)
  }
}

// === JSON configuration schema ===
case class CrossValidatingAlloyForgeConfig(version: String,
                                           forgeName: String,
                                           forgeConfig: JObject,
                                           numFolds: Int = 5,
                                           portion: Double = 1.0,
                                           foldSeed: Long = new Random().nextLong(),
                                           skipFinalTraining: Boolean = false)
