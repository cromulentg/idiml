package com.idibon.ml.train.datagenerator

import com.idibon.ml.common.EmbeddedEngine
import com.idibon.ml.test.BasicFeaturePipeline

import org.apache.spark.mllib.linalg.{Vector, Vectors}

import org.json4s.JObject
import org.json4s.JsonDSL._

import org.json4s.native.JsonMethods.parse
import org.scalatest.{FunSpec, Matchers}

import scala.io.Source

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
