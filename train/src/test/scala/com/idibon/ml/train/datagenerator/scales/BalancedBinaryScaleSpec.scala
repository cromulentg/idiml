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

  /**
    * Uses numSlices = 4 to always partition the data into four sets.
    * This is needed because by default, when run locally, it defaults
    * to partitioning to the number of cores. The sample method, as I
    * understand, samples on a per partition basis. Since testing
    * and the like happens on various machines, it would result in a
    * different amount of partitions (since the number of cores isn't
    * fixed) and so, with fluctuating partitions, you will get different
    * results and hence failing tests; therefore we hard code it.
    */
  describe("BalancedBinaryScale") {

    it("should balance a dataset with too many negatives") {
      val gen = new BalancedBinaryScaleBuilder(seed = 1L).build()
      val negatives = (0 until 10).map(x => LabeledPoint(0.0, Vectors.sparse(5, Array(1, 2, 3), Array(x.toFloat, 1.0, 1.0))))
      val positives = (0 until 2).map(x => LabeledPoint(1.0, Vectors.sparse(5, Array(1, 2, 3), Array(x.toFloat, 1.0, 1.0))))
      val actual = gen.balance("testLabel", engine.sparkContext.parallelize(negatives ++ positives, numSlices = 4))
      actual.count() shouldBe 5
      actual.filter(l => l.label == 0.0).count() shouldBe 3
      actual.filter(l => l.label == 1.0).count() shouldBe 2
    }

    it("should balance a dataset with too many positives") {
      val gen = new BalancedBinaryScaleBuilder(seed = 1L).build()
      val negatives = (0 until 2).map(x => LabeledPoint(0.0, Vectors.sparse(5, Array(1, 2, 3), Array(1.0, 1.0, 1.0))))
      val positives = (0 until 10).map(x => LabeledPoint(1.0, Vectors.sparse(5, Array(1, 2, 3), Array(1.0, 1.0, 1.0))))
      val actual = gen.balance("testLabel", engine.sparkContext.parallelize(negatives ++ positives, numSlices = 4))
      actual.count() shouldBe 5
      actual.filter(l => l.label == 0.0).count() shouldBe 2
      actual.filter(l => l.label == 1.0).count() shouldBe 3
    }

    it("should not do anything to a dataset with ratio inside threshold") {
      val gen = new BalancedBinaryScaleBuilder(seed = 1L).build()
      val negatives = (0 until 10).map(x => LabeledPoint(0.0, Vectors.sparse(5, Array(1, 2, 3), Array(1.0, 1.0, 1.0))))
      val positives = (0 until 7).map(x => LabeledPoint(1.0, Vectors.sparse(5, Array(1, 2, 3), Array(1.0, 1.0, 1.0))))
      val actual = gen.balance("testLabel", engine.sparkContext.parallelize(negatives ++ positives, numSlices = 4))
      actual.count() shouldBe 17
      actual.filter(l => l.label == 0.0).count() shouldBe 10
      actual.filter(l => l.label == 1.0).count() shouldBe 7
    }

    it("should not do anything to a dataset with wrong number of `classes`") {
      val gen = new BalancedBinaryScaleBuilder(seed = 1L).build()
      val negatives = (0 until 10).map(x => LabeledPoint(0.0, Vectors.sparse(5, Array(1, 2, 3), Array(1.0, 1.0, 1.0))))
      val positives1 = (0 until 7).map(x => LabeledPoint(1.0, Vectors.sparse(5, Array(1, 2, 3), Array(1.0, 1.0, 1.0))))
      val positives2 = (0 until 7).map(x => LabeledPoint(2.0, Vectors.sparse(5, Array(1, 2, 3), Array(1.0, 1.0, 1.0))))
      val actual = gen.balance(
        "testLabel",
        engine.sparkContext.parallelize(negatives ++ positives1 ++ positives2, numSlices = 4))
      actual.count() shouldBe 24
      actual.filter(l => l.label == 0.0).count() shouldBe 10
      actual.filter(l => l.label == 1.0).count() shouldBe 7
      actual.filter(l => l.label == 2.0).count() shouldBe 7
    }
  }
}

/**
  * Verifies the functionality of SingleClassDataSetDefender trait
  */
class SingleClassDataSetDefenderSpec extends FunSpec with Matchers
  with BeforeAndAfter with BeforeAndAfterAll {

  val engine = new EmbeddedEngine

  /**
    * Uses numSlices = 4 to always partition the data into four sets.
    * This is needed because by default, when run locally, it defaults
    * to partitioning to the number of cores. The sample method, as I
    * understand, samples on a per partition basis. Since testing
    * and the like happens on various machines, it would result in a
    * different amount of partitions (since the number of cores isn't
    * fixed) and so, with fluctuating partitions, you will get different
    * results and hence failing tests; therefore we hard code it.
    */
  describe("Defend tests") {

    it("should return None when we have more than 1 class.") {
      val gen = new DummyScaler()
      val negatives = (0 until 10).map(x => LabeledPoint(0.0, Vectors.sparse(5, Array(1, 2, 3), Array(x.toFloat, 1.0, 1.0))))
      val positives = (0 until 2).map(x => LabeledPoint(1.0, Vectors.sparse(5, Array(1, 2, 3), Array(x.toFloat, 1.0, 1.0))))
      val actual = gen.defend(engine.sparkContext.parallelize(negatives ++ positives, numSlices = 4))
      actual shouldBe None
    }

    it("should return Some with an equal number of opposite polarity items when passed in class have value 1.0") {
      val gen = new DummyScaler()
      val positives = (0 until 2).map(x => LabeledPoint(1.0, Vectors.sparse(5, Array(1, 2, 3), Array(x.toFloat, 1.0, 1.0))))
      val Some(actual) = gen.defend(engine.sparkContext.parallelize(positives, numSlices = 4))
      actual.count() shouldBe 4
      actual.filter(l => l.label == 0.0).count() shouldBe 2
      actual.filter(l => l.label == 1.0).count() shouldBe 2
    }

    it("should return Some with an equal number of opposite polarity items when passed in class have value 0.0") {
      val gen = new DummyScaler()
      val positives = (0 until 2).map(x => LabeledPoint(0.0, Vectors.sparse(5, Array(1, 2, 3), Array(x.toFloat, 1.0, 1.0))))
      val Some(actual) = gen.defend(engine.sparkContext.parallelize(positives, numSlices = 4))
      actual.count() shouldBe 4
      actual.filter(l => l.label == 0.0).count() shouldBe 2
      actual.filter(l => l.label == 1.0).count() shouldBe 2
    }

    it("should return Some with an equal number of opposite polarity items when passed in class have value 2.0") {
      val gen = new DummyScaler()
      val positives = (0 until 2).map(x => LabeledPoint(2.0, Vectors.sparse(5, Array(1, 2, 3), Array(x.toFloat, 1.0, 1.0))))
      val Some(actual) = gen.defend(engine.sparkContext.parallelize(positives, numSlices = 4))
      actual.count() shouldBe 4
      actual.filter(l => l.label == 2.0).count() shouldBe 2
      actual.filter(l => l.label == 1.0).count() shouldBe 2
    }
  }

  case class DummyScaler() extends SingleClassDataSetDefender
}
