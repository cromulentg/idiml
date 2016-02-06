package com.idibon.ml.predict.ml

import org.apache.spark.ml.classification.IdibonSparkLogisticRegressionModelWrapper
import org.apache.spark.mllib.linalg.Vectors
import org.scalatest.{ParallelTestExecution, BeforeAndAfter, Matchers, FunSpec}

/**
  * Class to test IdibonSparkLogisticRegressionModelWrapper methods.
  */
class IdibonSparkLogisticRegressionModelWrapperSpec extends FunSpec
with Matchers with BeforeAndAfter with ParallelTestExecution {

  val model = new IdibonSparkLogisticRegressionModelWrapper(
    "test", Vectors.sparse(10, Array(0, 3, 5), Array(0.3, 0.2, 0.1)), 0.0)

  describe("predict probability") {
    it("handles vectors of different sizes well when we need to prune.") {
      val result = model.predictProbability(Vectors.sparse(11, Array(0, 3, 11), Array(1.0, 1.0, 1.0)))
      result shouldBe Vectors.sparse(2, Array(0, 1), Array(0.3775406687981454,0.6224593312018546))
    }

    it("handles vectors of different sizes well") {
      val result = model.predictProbability(Vectors.sparse(11, Array(0, 3, 9), Array(1.0, 1.0, 0.0)))
      result shouldBe Vectors.sparse(2, Array(0, 1), Array(0.3775406687981454,0.6224593312018546))
    }

    it("handles vectors equal to model size") {
      val result = model.predictProbability(Vectors.sparse(10, Array(0, 3, 11), Array(1.0, 1.0, 1.0)))
      result shouldBe Vectors.sparse(2, Array(0, 1), Array(0.3775406687981454,0.6224593312018546))
    }
  }

  describe("get significant features") {
    it("handles getting no significant features when all features map to zero weight value") {
      val result = model.getSignificantDimensions(
        Vectors.sparse(10, Array(1, 2, 4), Array(1.0, 1.0, 1.0)), 0.5f)
      result.numActives shouldBe 0
    }
    it("handles getting no significant features when values are below the threshold") {
      val result = model.getSignificantDimensions(
        Vectors.sparse(10, Array(1, 2, 5), Array(1.0, 1.0, 1.0)), 0.90f)
      result.numActives shouldBe 0
    }
    it("handles getting significant features when values are above the threshold") {
      val result = model.getSignificantDimensions(
        Vectors.sparse(10, Array(0, 3, 5), Array(1.0, 1.0, 1.0)), 0.53f)
      result.toArray shouldBe Array(0.574442516811659, 0, 0, 0.549833997312478, 0, 0, 0, 0, 0, 0)
    }
  }
}
