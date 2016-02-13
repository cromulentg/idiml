package com.idibon.ml.train.datagenerator.scales

import com.idibon.ml.common.EmbeddedEngine
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.scalatest.{BeforeAndAfter, BeforeAndAfterAll, FunSpec, Matchers}

/**
  * Verifies the functionality of NoOpScale class
  */
class NoOpScaleSpec extends FunSpec with Matchers
  with BeforeAndAfter with BeforeAndAfterAll {

  val engine = new EmbeddedEngine

  describe("NoOpScale tests") {

    it("should not change anything passed in to it with too many negatives") {
      val gen = new NoOpScale()
      val negatives = (0 until 10).map(x => LabeledPoint(0.0, Vectors.sparse(5, Array(1, 2, 3), Array(x.toFloat, 1.0, 1.0))))
      val positives = (0 until 2).map(x => LabeledPoint(1.0, Vectors.sparse(5, Array(1, 2, 3), Array(x.toFloat, 1.0, 1.0))))
      val actual = gen.balance("testLabel", engine.sparkContext.parallelize(negatives ++ positives))
      actual.count() shouldBe 12
      actual.filter(l => l.label == 0.0).count() shouldBe 10
      actual.filter(l => l.label == 1.0).count() shouldBe 2
    }

    it("should not change anything passed in to it with too many positives") {
      val gen = new NoOpScale()
      val negatives = (0 until 2).map(x => LabeledPoint(0.0, Vectors.sparse(5, Array(1, 2, 3), Array(1.0, 1.0, 1.0))))
      val positives = (0 until 10).map(x => LabeledPoint(1.0, Vectors.sparse(5, Array(1, 2, 3), Array(1.0, 1.0, 1.0))))
      val actual = gen.balance("testLabel", engine.sparkContext.parallelize(negatives ++ positives))
      actual.count() shouldBe 12
      actual.filter(l => l.label == 0.0).count() shouldBe 2
      actual.filter(l => l.label == 1.0).count() shouldBe 10
    }

    it("sshould not change anything passed in to it with ratio inside threshold") {
      val gen = new NoOpScale()
      val negatives = (0 until 10).map(x => LabeledPoint(0.0, Vectors.sparse(5, Array(1, 2, 3), Array(1.0, 1.0, 1.0))))
      val positives = (0 until 7).map(x => LabeledPoint(1.0, Vectors.sparse(5, Array(1, 2, 3), Array(1.0, 1.0, 1.0))))
      val actual = gen.balance("testLabel", engine.sparkContext.parallelize(negatives ++ positives))
      actual.count() shouldBe 17
      actual.filter(l => l.label == 0.0).count() shouldBe 10
      actual.filter(l => l.label == 1.0).count() shouldBe 7
    }
  }
}
