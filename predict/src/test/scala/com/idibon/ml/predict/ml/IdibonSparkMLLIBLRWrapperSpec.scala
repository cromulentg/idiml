package com.idibon.ml.predict.ml

import org.apache.spark.mllib.classification.IdibonSparkMLLIBLRWrapper
import org.apache.spark.mllib.linalg.Vectors
import org.scalatest.{BeforeAndAfter, FunSpec, Matchers, ParallelTestExecution}

/**
  * Class to test IdibonSparkMLLIBLRWrapper methods.
  */
class IdibonSparkMLLIBLRWrapperSpec extends FunSpec
  with Matchers with BeforeAndAfter with ParallelTestExecution {

  val model = new IdibonSparkMLLIBLRWrapper(
    Vectors.sparse(10, Array(0, 3, 5), Array(0.6, 0.8, 0.1)).toDense, 0.0, 10, 2)
  val multinomialmodel = new IdibonSparkMLLIBLRWrapper(
    Vectors.sparse(20, Array(0, 3, 5, 10, 13, 15), Array(0.6, 0.8, 0.1, 0.3, 0.4, 0.05)).toDense, 0.0, 10, 3)

  describe("predict probability binary case") {
    it("handles vectors of different sizes well when we need to prune") {
      val result = model.predictProbability(Vectors.sparse(11, Array(0, 3, 11), Array(1.0, 1.0, 1.0)))
      result shouldBe Vectors.sparse(2, Array(0, 1), Array(0.1978161114414183,0.8021838885585817))
    }

    it("handles vectors of different sizes well") {
      val result = model.predictProbability(Vectors.sparse(11, Array(0, 3, 9), Array(1.0, 1.0, 0.0)))
      result shouldBe Vectors.sparse(2, Array(0, 1), Array(0.1978161114414183,0.8021838885585817))
    }

    it("handles vectors equal to model size") {
      val result = model.predictProbability(Vectors.sparse(10, Array(0, 3, 9), Array(1.0, 1.0, 1.0)))
      result shouldBe Vectors.sparse(2, Array(0, 1), Array(0.1978161114414183,0.8021838885585817))
    }
  }

  describe("predict probability n-ary case") {
    it("handles vectors of different sizes well") {
      val result = multinomialmodel.predictProbability(Vectors.sparse(11, Array(0, 3, 11), Array(1.0, 1.0, 1.0)))
      result shouldBe Vectors.sparse(3, Array(0,1,2), Array(0.16818777216816616, 0.5, 0.3318122278318339))
      result.toDense.values.sum shouldBe 1.0
    }

    it("handles vectors equal to model size") {
      val result = multinomialmodel.predictProbability(Vectors.sparse(10, Array(0, 3, 9), Array(1.0, 1.0, 1.0)))
      result shouldBe Vectors.sparse(3, Array(0,1,2), Array(0.16818777216816616, 0.5, 0.3318122278318339))
      result.toDense.values.sum shouldBe 1.0
    }
  }

  describe("get significant features") {
    it("handles getting no significant features when all features map to zero weight value") {
      val result = model.getSignificantFeatures(
        Vectors.sparse(10, Array(1, 2, 4), Array(1.0, 1.0, 1.0)), 0.55f)
      result.size shouldBe 2
      result(0) shouldBe (0, List())
      result(1) shouldBe (1, List())
    }
    it("handles getting no significant features when values are below the threshold") {
      val result = model.getSignificantFeatures(
        Vectors.sparse(10, Array(1, 2, 5), Array(1.0, 1.0, 1.0)), 0.90f)
      result.size shouldBe 2
      result(0) shouldBe (0, List())
      result(1) shouldBe (1, List())
    }
    it("handles getting significant features when values are above the threshold") {
      val result = model.getSignificantFeatures(
        Vectors.sparse(10, Array(0, 3, 5), Array(1.0, 1.0, 1.0)), 0.53f)
      result.size shouldBe 2
      result(0) shouldBe (0, List())
      result(1) shouldBe (1, List((0, 0.6456563f), (3, 0.6899745f)))
    }

    it("handles n-ary case of getting significant features") {
      val result = multinomialmodel.getSignificantFeatures(
        Vectors.sparse(10, Array(0, 3, 5), Array(1.0, 1.0, 1.0)), 0.50f)
      result.size shouldBe 3
      result(0) shouldBe (0, List())
      result(1) shouldBe (1, List((0, 0.5), (3, 0.5), (5, 0.5)))
      result(2) shouldBe (2, List())
    }
  }
}
