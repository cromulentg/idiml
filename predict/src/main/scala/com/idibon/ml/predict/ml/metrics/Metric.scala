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


