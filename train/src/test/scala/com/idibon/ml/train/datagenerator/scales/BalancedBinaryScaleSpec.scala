package com.idibon.ml.train.datagenerator.scales

import com.idibon.ml.common.EmbeddedEngine
import com.idibon.ml.feature.indexer.IndexTransformer
import com.idibon.ml.feature.{ContentExtractor, FeaturePipeline, FeaturePipelineBuilder}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.json4s.JObject
import org.json4s.native.JsonMethods.parse
import org.scalatest.{BeforeAndAfter, BeforeAndAfterAll, FunSpec, Matchers}

import scala.io.Source

/**
  * Verifies the functionality of BalancedBinaryScale class
  */
class BalancedBinaryScaleSpec extends FunSpec with Matchers
  with BeforeAndAfter with BeforeAndAfterAll {

  val engine = new EmbeddedEngine

  describe("BalancedBinaryScale") {

    it("should balance a dataset with too many negatives") {
      val gen = new BalancedBinaryScaleBuilder(seed = 1L).build()
      val negatives = (0 until 10).map(x => LabeledPoint(0.0, Vectors.sparse(5, Array(1, 2, 3), Array(x.toFloat, 1.0, 1.0))))
      val positives = (0 until 2).map(x => LabeledPoint(1.0, Vectors.sparse(5, Array(1, 2, 3), Array(x.toFloat, 1.0, 1.0))))
      val actual = gen.balance("testLabel", engine.sparkContext.parallelize(negatives ++ positives))
      actual.count() shouldBe 4
      actual.filter(l => l.label == 0.0).count() shouldBe 2
      actual.filter(l => l.label == 1.0).count() shouldBe 2
    }

    it("should balance a dataset with too many positives") {
      val gen = new BalancedBinaryScaleBuilder(seed = 1L).build()
      val negatives = (0 until 2).map(x => LabeledPoint(0.0, Vectors.sparse(5, Array(1, 2, 3), Array(1.0, 1.0, 1.0))))
      val positives = (0 until 10).map(x => LabeledPoint(1.0, Vectors.sparse(5, Array(1, 2, 3), Array(1.0, 1.0, 1.0))))
      val actual = gen.balance("testLabel", engine.sparkContext.parallelize(negatives ++ positives))
      actual.count() shouldBe 4
      actual.filter(l => l.label == 0.0).count() shouldBe 2
      actual.filter(l => l.label == 1.0).count() shouldBe 2
    }

    it("should not do anything to a dataset with ratio inside threshold") {
      val gen = new BalancedBinaryScaleBuilder(seed = 1L).build()
      val negatives = (0 until 10).map(x => LabeledPoint(0.0, Vectors.sparse(5, Array(1, 2, 3), Array(1.0, 1.0, 1.0))))
      val positives = (0 until 7).map(x => LabeledPoint(1.0, Vectors.sparse(5, Array(1, 2, 3), Array(1.0, 1.0, 1.0))))
      val actual = gen.balance("testLabel", engine.sparkContext.parallelize(negatives ++ positives))
      actual.count() shouldBe 17
      actual.filter(l => l.label == 0.0).count() shouldBe 10
      actual.filter(l => l.label == 1.0).count() shouldBe 7
    }

    it("should not do anything to a dataset with wrong number of `classes`") {
      val gen = new BalancedBinaryScaleBuilder(seed = 1L).build()
      val negatives = (0 until 10).map(x => LabeledPoint(0.0, Vectors.sparse(5, Array(1, 2, 3), Array(1.0, 1.0, 1.0))))
      val actual = gen.balance("testLabel", engine.sparkContext.parallelize(negatives))
      actual.count() shouldBe 10
      actual.filter(l => l.label == 0.0).count() shouldBe 10
    }
  }
}
