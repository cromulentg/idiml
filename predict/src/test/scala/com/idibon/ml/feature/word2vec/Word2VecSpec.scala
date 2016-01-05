package com.idibon.ml.feature.word2vec

/**
  *
  * Code to deal with SparkContext taken from
  * https://github.com/apache/spark/blob/master/mllib/src/test/scala/org/apache/spark/mllib/util/MLlibTestSparkContext.scala
  *
  */

import org.apache.spark.mllib.linalg._
import org.apache.spark.{SparkConf, SparkContext}

import org.scalatest._

trait Word2VecSparkContext extends BeforeAndAfterAll { self: Suite =>
  @transient var sc: SparkContext = _
  @transient var transform: Word2VecTransformer = _

  override def beforeAll() {
    super.beforeAll()
    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("Word2VecSpec")
    sc = new SparkContext(conf)
    transform = new Word2VecTransformer(sc,"src/test/resources/fixtures/model")
  }

  override def afterAll() {
    if (sc != null) {
      sc.stop()
    }
    sc = null
    super.afterAll()
  }
}

class Word2VecSpec extends FunSpec with Matchers with Word2VecSparkContext {

  describe("Word2Vec") {

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