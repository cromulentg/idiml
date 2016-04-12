package com.idibon.ml.train.datagenerator

import com.idibon.ml.common.EmbeddedEngine
import com.idibon.ml.test.BasicFeaturePipeline

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.json4s.JObject
import org.json4s.native.JsonMethods.parse
import org.scalatest.{FunSpec, Matchers}

import scala.io.Source

/** Verifies the functionality of MultiClassDataFrameGenerator
  *
  */
class MultiClassDataFrameGeneratorSpec extends FunSpec with Matchers {

  val documents = {
    implicit val formats = org.json4s.DefaultFormats
    val path = getClass.getClassLoader
      .getResource("test_data/multiclass_labeled_points.json").getPath
    Source.fromFile(path).getLines.map(l => parse(l)).map(_.extract[JObject]).toList
  }

  val pipeline = {
    val unprimed = BasicFeaturePipeline.classification
    unprimed.prime(documents)
  }

  val engine = new EmbeddedEngine

  it("should create a single model with all positive labels") {
    val gen = new MultiClassDataFrameGeneratorBuilder().build()
    val models = gen(engine, pipeline, () => documents).sortBy(_.id)

    models.head.labels.keys should contain theSameElementsAs List("Intent to Buy", "Monkey")

    val points = models.head.frame.collect
      .map(r => LabeledPoint(r.getAs[Double]("label"), r.getAs[Vector]("features")))

    val monkey = points.find(_.label == models.head.labels("Monkey")).get
    val intent = points.find(_.label == models.head.labels("Intent to Buy")).get

    monkey.label match {
      case 0.0 =>
        monkey shouldBe LabeledPoint(0.0, Vectors.dense(1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        intent shouldBe LabeledPoint(1.0, Vectors.dense(0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1))
      case 1.0 =>
        intent shouldBe LabeledPoint(0.0, Vectors.dense(1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0))
        monkey shouldBe LabeledPoint(1.0, Vectors.dense(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1))
    }
  }
}
