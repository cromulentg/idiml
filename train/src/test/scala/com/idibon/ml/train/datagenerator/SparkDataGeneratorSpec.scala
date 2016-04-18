package com.idibon.ml.train.datagenerator

import com.idibon.ml.common.EmbeddedEngine
import com.idibon.ml.test.BasicFeaturePipeline

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.json4s.JObject
import org.json4s.JsonDSL._
import org.json4s.native.JsonMethods.parse
import org.scalatest.{FunSpec, Matchers}

import scala.io.Source

class SparkDataGeneratorSpec extends FunSpec with Matchers {

  describe("deepMerge") {

    it("should return an empty map as the sum of two empty maps") {
      SparkDataGenerator.deepMerge(Map(), Map()) shouldBe empty
    }

    it("should merge the value hash of identical keys") {
      val a = Map("a" -> Map("b" -> 0.0))
      val b = Map("a" -> Map("c" -> 1.0))
      SparkDataGenerator.deepMerge(a, b) shouldBe Map("a" -> Map("b" -> 0.0, "c" -> 1.0))
    }

    it("should produce the union of all outer keys") {
      val a = Map("a" -> Map[String, Double]())
      val b = Map("b" -> Map[String, Double]())
      SparkDataGenerator.deepMerge(a, b) shouldBe Map("a" -> Map(), "b" -> Map())
    }
  }
}

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

  it("should build datasets incrementally") {
    val gen = new SparkDataGenerator(scales.NoOpScale(),
      new MulticlassLabeledPointGenerator, 1) {}

    val model = gen(engine, pipeline, () => documents ++ documents).head
    model.labels.keys should contain theSameElementsAs List("Intent to Buy", "Monkey")
    model.frame.count shouldBe 4
  }

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
        monkey shouldBe LabeledPoint(0.0, Vectors.dense(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1))
        intent shouldBe LabeledPoint(1.0, Vectors.dense(1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0))
      case 1.0 =>
        intent shouldBe LabeledPoint(0.0, Vectors.dense(1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0))
        monkey shouldBe LabeledPoint(1.0, Vectors.dense(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1))
    }
  }
}

/** Verifies the functionality of KClassDataFrameGenerator
  *
  */
class KClassDataFrameGeneratorSpec extends FunSpec with Matchers {

  val documents = {
    implicit val formats = org.json4s.DefaultFormats
    val path = getClass.getClassLoader
      .getResource("test_data/labeled_points.json").getPath
    Source.fromFile(path).getLines.map(l => parse(l)).map(_.extract[JObject]).toList
  }

  val pipeline = {
    val unprimed = BasicFeaturePipeline.classification
    unprimed.prime(documents)
  }

  val engine = new EmbeddedEngine

  it("should generate a dataframe for each label") {
    val docs = List[JObject](
      ("content" -> "Would you recommend a chevy malibu?") ~
        ("annotations" -> List(
          ("label" -> ("name" -> "Desire")) ~ ("isPositive" -> true),
          ("label" -> ("name" -> "NoDesire")) ~ ("isPositive" -> false))),
      ("content" -> "Who drives it?") ~
        ("annotations" -> List(
          ("label" -> ("name" -> "Desire")) ~ ("isPositive" -> false),
          ("label" -> ("name" -> "NoDesire")) ~ ("isPositive" -> true))))

    val gen = new KClassDataFrameGeneratorBuilder().build
    val models = gen(engine, pipeline, () => docs).sortBy(_.id)

    models.map(_.id) shouldBe List("Desire", "NoDesire")
    models.head.labels shouldBe Map("positive" -> 1.0, "negative" -> 0.0)
    models.head.frame.collect.map(_.getAs[Double]("label")) shouldBe List(1.0, 0.0)
    models.head.frame.collect.map(_.getAs[Vector]("features")) shouldBe List(
      Vectors.dense(0, 0, 1, 1, 1, 1, 1, 1, 0), Vectors.dense(1, 1, 0, 0, 0, 0, 0, 0, 1))
  }
}
