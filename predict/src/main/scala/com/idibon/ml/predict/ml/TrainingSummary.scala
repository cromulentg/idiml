package com.idibon.ml.predict.ml

import com.idibon.ml.alloy.Codec
import com.idibon.ml.feature.{FeatureOutputStream, FeatureInputStream, Builder, Buildable}
import com.idibon.ml.predict.ml.metrics.MetricClass.MetricClass
import com.idibon.ml.predict.ml.metrics._

/**
  * Class that wraps results for an individual PredictModel.
 *
  * @param identifier some identifier, usually a label in our case.
  * @param metrics a sequence of metrics that we've managed to gather.
  */
case class TrainingSummary(identifier: String, metrics: Seq[Metric with Buildable[_, _]])
  extends Buildable[TrainingSummary, TrainingSummaryBuilder] {
  /** Stores the data to an output stream so it may be reloaded later.
    *
    * @param output - Output stream to write data
    */
  override def save(output: FeatureOutputStream): Unit = {
    Codec.String.write(output, identifier)
    Codec.VLuint.write(output, metrics.size)
    metrics.foreach { m => output.writeBuildable(m) }
  }

  override def toString: String = {
    val sb = new StringBuilder(s"----- Training summary for $identifier ---- Total = ${metrics.size}\n")
    metrics.sortBy(m => (m.metricType, m.metricClass)).foreach(m => {
      sb.append(m.toString()).append("\n")
    })
    sb.mkString
  }
}

object TrainingSummary {

  /**
    * Averages a bunch of training summaries.
    *
    * Groups metrics by type and uses the static metric average function to average them.
    *
    * @param name the name to give the new summary
    * @param summaries the sequence of training summaries to get metrics from to average.
    * @param mClass the metric class to give the new metrics.
    * @return a training summary with "averaged" metrics
    */
  def averageSummaries(name: String, summaries: Seq[TrainingSummary], mClass: MetricClass) = {
    val allMetrics = summaries.flatMap(ts => ts.metrics)
    val groupedMetrics = allMetrics.groupBy(m => m.metricType)
    val averagedMetrics = groupedMetrics.flatMap({ case (metricType, metrics) =>
      Metric.average(metrics, Some(mClass))
    }).toSeq
    new TrainingSummary(name, averagedMetrics)
  }
}

/**
  * Class to build training summaries.
  */
class TrainingSummaryBuilder extends Builder[TrainingSummary] {
  /** Instantiates and loads an object from an input stream
    *
    * @param input - Data stream where object was previously saved
    */
  override def build(input: FeatureInputStream): TrainingSummary = {
    val label = Codec.String.read(input)
    val size = Codec.VLuint.read(input)
    new TrainingSummary(label,
      (0 until size).map(_ => input.readBuildable.asInstanceOf[Metric with Buildable[_, _]]))
  }
}
