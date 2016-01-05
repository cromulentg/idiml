package com.idibon.ml.feature.word2vec

import org.apache.spark.mllib.linalg._
import com.idibon.ml.test.Spark

import org.scalatest.{FunSpec, Matchers}

class Word2VecSpec extends FunSpec with Matchers {

  describe("Word2Vec") {

    val transform = new Word2VecTransformer(Spark.sc,"src/test/resources/fixtures/model")

    it("should return an empty sparse vector of size 100 on empty input") {
      val expected = Vectors.sparse(100, Array.empty[Int], Array.empty[Double])
      transform(Seq[String]()) shouldBe expected
    }

    it("should return a vector of non-zero floats with a single word input"){
      val oneWordSeq = Seq[String]("anarchist")
      val outputVector = transform(oneWordSeq)
      outputVector should have size 100
      for (elem <- outputVector.toArray)
        elem should (be > -1.0 and be < 1.0 and not be 0)
    }

    it("should return a vector of non-zero floats with a multi-word input"){
      val multiwordSeq = Seq[String]("kropotkin","is","considered","an", "anarchist")
      val outputVector = transform(multiwordSeq).toArray
      outputVector should have size 100
      for (elem <- outputVector.toArray)
        elem should (be > -1.0 and be < 1.0 and not be 0)
    }

    it("should raise an exception on out-of-vocabulary input"){
      val oneWordSeq = Seq[String]("batman")
      val thrown = intercept[IllegalStateException] {
        transform(oneWordSeq).toArray
      }
      assert(thrown.getMessage === "batman not in vocabulary")
    }
  }
}