package com.idibon.ml.train.datagenerator

import scala.io.Source

import com.idibon.ml.test.BasicFeaturePipeline

import org.json4s.JObject
import org.json4s.JsonDSL._
import org.json4s.native.JsonMethods.parse
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

import org.scalatest.{FunSpec, Matchers}

class KClassLabeledPointGeneratorSpec extends FunSpec with Matchers {
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

  it("should return negative training items") {
    val lpg = new KClassLabeledPointGenerator
    val doc = ("content" -> "Would you recommend a chevy malibu?") ~
      ("annotations" -> List(
        ("label" -> ("name" -> "Foo")) ~ ("isPositive" -> true),
        ("label" -> ("name" -> "Bar")) ~ ("isPositive" -> false)))
    lpg(pipeline, doc) shouldBe List(
      TrainingPoint("Foo", "positive", LabeledPoint(1.0,
        Vectors.dense(0, 0, 1, 1, 1, 1, 1, 1, 0))),
      TrainingPoint("Bar", "negative", LabeledPoint(0.0,
        Vectors.dense(0, 0, 1, 1, 1, 1, 1, 1, 0))))
  }

  it("should return an empty list if there are no annotations") {
    val lpg = new KClassLabeledPointGenerator
    val doc = ("content" -> "Who recommend drives?") ~ ("annotations" -> List[JObject]())
    lpg(pipeline, doc) shouldBe empty
  }

  it("should raise an exception for span annotations") {
    val lpg = new KClassLabeledPointGenerator
    val doc = ("content" -> "Who drives it?") ~
      ("annotations" -> List(
        ("label" -> ("name" -> "Foo")) ~ ("isPositive" -> true) ~
          ("offset" -> 0) ~ ("length" -> 1)))
    intercept[IllegalArgumentException] { lpg(pipeline, doc) }
  }
}

class MulticlassLabeledPointGeneratorSpec extends FunSpec with Matchers {

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

  it("should assign label IDs in the order they are observed") {
    val lpg1 = new MulticlassLabeledPointGenerator
    val pts1 = documents.flatMap(doc => lpg1(pipeline, doc))
    val lpg2 = new MulticlassLabeledPointGenerator
    val pts2 = documents.reverse.flatMap(doc => lpg2(pipeline, doc)).reverse

    pts1.map(p => (p.labelName, p.p.label)) shouldBe List("Intent to Buy" -> 0.0, "Monkey" -> 1.0)
    pts2.map(p => (p.labelName, p.p.label)) shouldBe List("Intent to Buy" -> 1.0, "Monkey" -> 0.0)
  }


  it("should return training points for positive annotations") {
    val lpg = new MulticlassLabeledPointGenerator

    val doc = ("content" -> "I like to recommend bananas") ~
      ("annotations" -> List(
        ("label" -> ("name" -> "Monkey")) ~ ("isPositive" -> true),
        ("label" -> ("name" -> "Intent to Buy")) ~ ("isPositive" -> true)))

    lpg(pipeline, doc) shouldBe List(
      TrainingPoint("model", "Monkey", LabeledPoint(0.0,
        Vectors.dense(0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1))),
      TrainingPoint("model", "Intent to Buy", LabeledPoint(1.0,
        Vectors.dense(0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1))))
  }

  it("should return an empty list if the document has 0 positive annotations") {
    val lpg = new MulticlassLabeledPointGenerator

    val doc = ("content" -> "I like to recommend bananas") ~
      ("annotations" -> List(
        ("label" -> ("name" -> "Monkey")) ~ ("isPositive" -> false),
        ("label" -> ("name" -> "Intent to Buy")) ~ ("isPositive" -> false)))

    lpg(pipeline, doc) shouldBe empty
  }

  it("should raise an exception for span annotations") {
    val lpg = new MulticlassLabeledPointGenerator

    val doc = ("content" -> "I like to recommend bananas") ~
      ("annotations" -> List(
        ("label" -> ("name" -> "Monkey")) ~ ("isPositive" -> false) ~
          ("offset" -> 0) ~ ("length" -> 0)))

    intercept[IllegalArgumentException] { lpg(pipeline, doc) }
  }

}
