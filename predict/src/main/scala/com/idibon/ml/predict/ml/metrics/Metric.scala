package com.idibon.ml.predict.ml.metrics

import com.idibon.ml.alloy.Codec
import com.idibon.ml.feature.{FeatureInputStream, FeatureOutputStream, Builder, Buildable}


/*
 * @author "Stefan Krawczyk <stefan@idibon.com>" on 2/10/16.
 *
 * These classes here encapsulate the type of data a metric
 * constitutes. Therefore only add new classes here if we
 * haven't got that data type covered.
 */

/**
  * Metric trait. All metrics extend this bad boy.
  */
trait Metric {
  val metricType: MetricTypes
  val metricClass: MetricClass.Value
}

/**
  * Base metric class that enforces that the metric type matches the data type we expect.
  * @param metricType
  * @param metricClass
  */
abstract class RawMetric(override val metricType: MetricTypes,
                         override val metricClass: MetricClass.Value) extends Metric {
  if (this.getClass != metricType.dataType)
    throw new IllegalArgumentException("Invalid data type!")
}

/**
  * Creates metrics that are around a float value.
  * @param mType
  * @param mClass
  * @param float
  */
case class FloatMetric(mType: MetricTypes,
                       mClass: MetricClass.Value,
                       float: Float)
  extends RawMetric(mType, mClass) with Buildable[FloatMetric, FloatMetricBuilder] {
  override def save(output: FeatureOutputStream): Unit = {
    Codec.String.write(output, metricType.toString)
    Codec.String.write(output, metricClass.toString)
    output.writeFloat(float)
  }
}
class FloatMetricBuilder extends Builder[FloatMetric] {
  override def build(input: FeatureInputStream): FloatMetric = {
    val mType = Codec.String.read(input)
    val mClass = Codec.String.read(input)
    val float = input.readFloat()
    new FloatMetric(MetricTypes.valueOf(mType), MetricClass.withName(mClass), float)
  }
}
object FloatMetric {
  /**
    * Averages a sequence of float metrics.
    * Assumes they are all the same class and type.
    * @param metrics the metrics to average.
    * @return a single averaged float metric
    */
  def average(metrics: Seq[FloatMetric]): FloatMetric = {
    val total = metrics.map(m => m.float).sum
    val avg = total / metrics.size.toFloat
    new FloatMetric(metrics.head.mType, metrics.head.mClass, avg)
  }
}

/**
  * Creates metrics that are around a float value & label.
  * @param mType
  * @param mClass
  * @param label
  * @param float
  */
case class LabelFloatMetric(mType: MetricTypes,
                            mClass: MetricClass.Value,
                            label: String,
                            float: Float)
  extends RawMetric(mType, mClass) with Buildable[LabelFloatMetric, LabelFloatMetricBuilder] {
  override def save(output: FeatureOutputStream): Unit = {
    Codec.String.write(output, metricType.toString)
    Codec.String.write(output, metricClass.toString)
    Codec.String.write(output, label)
    output.writeFloat(float)
  }
}
class LabelFloatMetricBuilder extends Builder[LabelFloatMetric] {
  override def build(input: FeatureInputStream): LabelFloatMetric = {
    val mType = Codec.String.read(input)
    val mClass = Codec.String.read(input)
    val label = Codec.String.read(input)
    val float = input.readFloat()
    new LabelFloatMetric(MetricTypes.valueOf(mType), MetricClass.withName(mClass), label, float)
  }
}
object LabelFloatMetric {
  /**
    * Averages a sequence of label float metrics based on label.
    * Assumes they are all the same class and type.
    * @param metrics the metrics to average.
    * @return a sequence of averaged label float metrics, one for each label.
    */
  def average(metrics: Seq[LabelFloatMetric]): Seq[LabelFloatMetric] = {
    val byLabel = metrics.groupBy(m => m.label)
    byLabel.map({case (label, labelMetrics) =>
      val total = labelMetrics.map(m => m.float).sum
      val avg = total / labelMetrics.size.toFloat
      new LabelFloatMetric(labelMetrics.head.mType, labelMetrics.head.mClass, labelMetrics.head.label, avg)
    }).toSeq
  }
}

/**
  * Creates metrics that are around an integer value.
  * @param mType
  * @param mClass
  * @param label
  * @param int
  */
case class LabelIntMetric(mType: MetricTypes,
                          mClass: MetricClass.Value,
                          label: String,
                          int: Int)
  extends RawMetric(mType, mClass) with Buildable[LabelIntMetric, LabelIntMetricBuilder] {
  override def save(output: FeatureOutputStream): Unit = {
    Codec.String.write(output, metricType.toString)
    Codec.String.write(output, metricClass.toString)
    Codec.String.write(output, label)
    Codec.VLuint.write(output, int)
  }
}
class LabelIntMetricBuilder extends Builder[LabelIntMetric] {
  override def build(input: FeatureInputStream): LabelIntMetric = {
    val mType = Codec.String.read(input)
    val mClass = Codec.String.read(input)
    val label = Codec.String.read(input)
    val int = Codec.VLuint.read(input)
    new LabelIntMetric(MetricTypes.valueOf(mType), MetricClass.withName(mClass), label, int)
  }
}
object LabelIntMetric {
  /**
    * Averages a sequence of label int metrics by label.
    * Assumes they are all the same class and type.
    * @param metrics the metrics to average.
    * @return a sequence of label int metrics, one per label.
    */
  def average(metrics: Seq[LabelIntMetric]): Seq[LabelIntMetric] = {
    val byLabel = metrics.groupBy(m => m.label)
    byLabel.map({case (label, labelMetrics) =>
      val total = labelMetrics.map(m => m.int).sum
      val avg = total.toFloat / labelMetrics.size.toFloat
      new LabelIntMetric(labelMetrics.head.mType, labelMetrics.head.mClass, labelMetrics.head.label, avg.toInt)
    }).toSeq
  }
}

/**
  * Creates metrics that are around a set of points. E.g. this could be plotted.
  * @param mType
  * @param mClass
  * @param points
  */
case class PointsMetric(mType: MetricTypes,
                        mClass: MetricClass.Value,
                        points: Seq[(Float, Float)])
  extends RawMetric(mType, mClass) with Buildable[PointsMetric, PointsMetricBuilder] {
  override def save(output: FeatureOutputStream): Unit = {
    Codec.String.write(output, metricType.toString)
    Codec.String.write(output, metricClass.toString)
    Codec.VLuint.write(output, points.size)
    points.foreach { case (x, y) => {
      output.writeFloat(x)
      output.writeFloat(y)
    }}
  }
}
class PointsMetricBuilder extends Builder[PointsMetric] {
  override def build(input: FeatureInputStream): PointsMetric = {
    val mType = Codec.String.read(input)
    val mClass = Codec.String.read(input)
    val size = Codec.VLuint.read(input)
    val points = (0 until size).map(_ => {
      (input.readFloat(), input.readFloat())
    })
    new PointsMetric(MetricTypes.valueOf(mType), MetricClass.withName(mClass), points)
  }
}
object PointsMetric {
  /**
    * Averages a sequence of points metrics.
    * Assumes they are all the same class and type and have the same X values.
    * @param metrics the metrics to average.
    * @return a single averaged points metric
    */
  def average(metrics: Seq[PointsMetric]): PointsMetric = {
    val byX = metrics.flatMap(x => x.points).groupBy(x => x._1)
    val points = byX.map({case (x, yMetrics) =>
      val total = yMetrics.map({case (xs, ys) => ys}).sum
      val avg = total / yMetrics.size.toFloat
      (x, avg)
    }).toSeq
    new PointsMetric(metrics.head.mType, metrics.head.mClass, points)
  }
}

/**
  * Creates metrics that are around a set of points for a label. E.g. this could be plotted.
  * @param mType
  * @param mClass
  * @param label
  * @param points
  */
case class LabelPointsMetric(mType: MetricTypes,
                             mClass: MetricClass.Value,
                             label: String,
                             points: Seq[(Float, Float)])
  extends RawMetric(mType, mClass) with Buildable[LabelPointsMetric, LabelPointsMetricBuilder] {
  override def save(output: FeatureOutputStream): Unit = {
    Codec.String.write(output, metricType.toString)
    Codec.String.write(output, metricClass.toString)
    Codec.String.write(output, label)
    Codec.VLuint.write(output, points.size)
    points.foreach { case (x, y) => {
      output.writeFloat(x)
      output.writeFloat(y)
    }}
  }
}
class LabelPointsMetricBuilder extends Builder[LabelPointsMetric] {
  override def build(input: FeatureInputStream): LabelPointsMetric = {
    val mType = Codec.String.read(input)
    val mClass = Codec.String.read(input)
    val label = Codec.String.read(input)
    val size = Codec.VLuint.read(input)
    val points = (0 until size).map(_ => {
      (input.readFloat(), input.readFloat())
    })
    new LabelPointsMetric(MetricTypes.valueOf(mType), MetricClass.withName(mClass), label, points)
  }
}
object LabelPointsMetric {
  /**
    * Averages a sequence of label points metrics by label.
    * Assumes they are all the same class and type and each label has the same X values.
    * @param metrics the metrics to average.
    * @return a sequence of averaged label points metrics, one for each label
    */
  def average(metrics: Seq[LabelPointsMetric]): Seq[LabelPointsMetric] = {
    val byLabel = metrics.groupBy(m => m.label)
    byLabel.map({case (label, labelMetrics) =>
      val byX = labelMetrics.flatMap(x => x.points).groupBy(x => x._1)
      val points = byX.map({case (x, yMetrics) =>
        val total = yMetrics.map({case (xs, ys) => ys}).sum
        val avg = total / yMetrics.size.toFloat
        (x, avg)
      }).toSeq
      new LabelPointsMetric(labelMetrics.head.mType, labelMetrics.head.mClass, labelMetrics.head.label, points)
    }).toSeq
  }
}

/**
  * Creates metrics that are around a set of properties.
  *
  * Makes all values strings. It is the consumers responsibility to
  * know how to reinterpret the string values.
  *
  * @param mType
  * @param mClass
  * @param properties
  */
case class PropertyMetric(mType: MetricTypes,
                          mClass: MetricClass.Value,
                          properties: Seq[(String, String)])
  extends RawMetric(mType, mClass) with Buildable[PropertyMetric, PropertyMetricBuilder] {
  override def save(output: FeatureOutputStream): Unit = {
    Codec.String.write(output, metricType.toString)
    Codec.String.write(output, metricClass.toString)
    Codec.VLuint.write(output, properties.size)
    properties.foreach { case (x, y) => {
      Codec.String.write(output, x)
      Codec.String.write(output, y)
    }}
  }
}
class PropertyMetricBuilder extends Builder[PropertyMetric] {
  override def build(input: FeatureInputStream): PropertyMetric = {
    val mType = Codec.String.read(input)
    val mClass = Codec.String.read(input)
    val size = Codec.VLuint.read(input)
    val properties = (0 until size).map(_ => {
      (Codec.String.read(input), Codec.String.read(input))
    })
    new PropertyMetric(MetricTypes.valueOf(mType), MetricClass.withName(mClass), properties)
  }
}


/**
  * Creates metrics that represent a confusion matrix.
  * @param mType
  * @param mClass
  * @param points
  */
case class ConfusionMatrixMetric(mType: MetricTypes,
                                 mClass: MetricClass.Value,
                                 points: Seq[(String, String, Float)])
  extends RawMetric(mType, mClass) with Buildable[ConfusionMatrixMetric, ConfusionMatrixMetricBuilder] {
  override def save(output: FeatureOutputStream): Unit = {
    Codec.String.write(output, metricType.toString)
    Codec.String.write(output, metricClass.toString)
    Codec.VLuint.write(output, points.size)
    points.foreach { case (x, y, z) => {
      Codec.String.write(output, x)
      Codec.String.write(output, y)
      output.writeFloat(z)
    }}
  }
}
class ConfusionMatrixMetricBuilder extends Builder[ConfusionMatrixMetric] {
  override def build(input: FeatureInputStream): ConfusionMatrixMetric = {
    val mType = Codec.String.read(input)
    val mClass = Codec.String.read(input)
    val size = Codec.VLuint.read(input)
    val points = (0 until size).map(_ => {
      (Codec.String.read(input), Codec.String.read(input), input.readFloat())
    })
    new ConfusionMatrixMetric(MetricTypes.valueOf(mType), MetricClass.withName(mClass), points)
  }
}
object ConfusionMatrixMetric {
  /**
    * Averages a sequence of confusion matrix metrics.
    * Assumes they are all the same class and type and have the same X, Y values.
    * @param metrics the metrics to average.
    * @return a single averaged confusion matrix metric.
    */
  def average(metrics: Seq[ConfusionMatrixMetric]): ConfusionMatrixMetric = {
    val byXY = metrics.flatMap(x => x.points).groupBy(x => (x._1, x._2))
    val points = byXY.map({case ((x, y), zMetrics) =>
      val total = zMetrics.map({case (xs, ys, zs) => zs}).sum
      val avg = total / zMetrics.size.toFloat
      (x, y, avg)
    }).toSeq
    new ConfusionMatrixMetric(metrics.head.mType, metrics.head.mClass, points)
  }
}

object Metric {
  /**
    * Static function to average a sequence of homogeneous metrics.
    *
    * Returns a sequence since some per label metrics will return an averaged metric per label.
    *
    * Throws assertion errors if the sequence is not homogeneous:
    *  - they need to be of the same RawMetric subclass
    *  - the same metric type
    *  - the same metric class
    *
    * @param metrics homogeneous metrics to average.
    * @return a sequence of averaged metrics.
    */
  def average(metrics: Seq[Metric with Buildable[_, _]]): Seq[Metric with Buildable[_, _]] = {
    val headMetricSubclass = metrics.head.getClass.getName
    val headMetricType = metrics.head.metricType
    val headMetricClass = metrics.head.metricClass
    assert(metrics.forall(x => x.getClass.getName.equals(headMetricSubclass)), "all metrics must be of same raw metric sub class")
    assert(metrics.forall(x => x.metricType == headMetricType), "all metrics must be of same metric type")
    assert(metrics.forall(x => x.metricClass == headMetricClass), "all metrics must be of same metric class")

    metrics.head match {
      case m: FloatMetric => Seq(FloatMetric.average(metrics.asInstanceOf[Seq[FloatMetric]]))
      case m: LabelFloatMetric => LabelFloatMetric.average(metrics.asInstanceOf[Seq[LabelFloatMetric]])
      case m: LabelIntMetric => LabelIntMetric.average(metrics.asInstanceOf[Seq[LabelIntMetric]])
      case m: PointsMetric => Seq(PointsMetric.average(metrics.asInstanceOf[Seq[PointsMetric]]))
      case m: LabelPointsMetric => LabelPointsMetric.average(metrics.asInstanceOf[Seq[LabelPointsMetric]])
      case m: PropertyMetric => metrics // no average defined
      case m: ConfusionMatrixMetric => Seq(ConfusionMatrixMetric.average(metrics.asInstanceOf[Seq[ConfusionMatrixMetric]]))
    }
  }
}
