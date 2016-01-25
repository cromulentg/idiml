package com.idibon.ml.feature

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.json4s.native.JsonMethods.parse
import org.json4s.{JObject, JDouble}

import org.scalatest.{Matchers, FunSpec}

class FeaturePipelineBuilderSpec extends FunSpec with Matchers {

  it("should support building pipelines with multiple stages and outputs") {

    val pipeline = (FeaturePipelineBuilder.named("test")
      += (FeaturePipelineBuilder.entry("A", new TransformA, "$document"))
      += (FeaturePipelineBuilder.entry("B", new TransformB, "$document"))
      := ("A", "B"))

    val document = parse("""{"A":0.375,"B":-0.625}""").asInstanceOf[JObject]
    val fp = pipeline.prime(List(document))
    fp(document) shouldBe Vectors.sparse(2, Array(0,1), Array(0.375,-0.625))
  }

  it("should support chained transforms") {
    val pipeline = (FeaturePipelineBuilder.named("test")
      += FeaturePipelineBuilder.entry("A", new TransformA, "$document")
      += FeaturePipelineBuilder.entry("C", new TransformC, "A")
      := ("C"))
    val document = parse("""{"A":-0.5}""").asInstanceOf[JObject]
    val fp = pipeline.prime(List(document))
    fp(document) shouldBe Vectors.sparse(1, Array(0), Array(0.5))
  }
}

private[this] class TransformA extends FeatureTransformer with TerminableTransformer {
  def apply(input: JObject): Vector = {
    Vectors.dense((input \ "A").asInstanceOf[JDouble].num)
  }
  override def numDimensions: Int = 1
  override def prune(transform: (Int) => Boolean): Unit = ???
  override def getHumanReadableFeature(indexes: Set[Int]): List[(Int, String)] = ???
  override def freeze(): Unit = {}
}

private[this] class TransformB extends FeatureTransformer with TerminableTransformer {
  def apply(input: JObject): Vector = {
    Vectors.dense((input \ "B").asInstanceOf[JDouble].num)
  }
  override def numDimensions: Int = 1
  override def prune(transform: (Int) => Boolean): Unit = ???
  override def getHumanReadableFeature(indexes: Set[Int]): List[(Int, String)] = ???
  override def freeze(): Unit = {}
}

private[this] class TransformC extends FeatureTransformer with TerminableTransformer {
  def apply(input: Vector): Vector = {
    Vectors.dense(input.toArray.map(_ + 1.0))
  }
  override def numDimensions: Int = 1
  override def prune(transform: (Int) => Boolean): Unit = ???
  override def getHumanReadableFeature(indexes: Set[Int]): List[(Int, String)] = ???
  override def freeze(): Unit = {}
}
