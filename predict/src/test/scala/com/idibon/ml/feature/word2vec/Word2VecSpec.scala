package com.idibon.ml.feature.word2vec

import com.idibon.ml.alloy.IntentAlloy
import com.idibon.ml.predict.EmbeddedEngine
import org.apache.spark.mllib.linalg._
import org.json4s._

import org.scalatest._

class Word2VecSpec extends FunSpec with Matchers {

  describe("Word2Vec") {

    val intentAlloy = new IntentAlloy()
    val path = "src/test/resources/fixtures/model"
    val config = JObject(JField("path",JString(path)))

    def loadModel(): Word2VecTransformer = {
      val loader = new Word2VecTransformerLoader()
      val transform = loader.load(new EmbeddedEngine, intentAlloy.reader, Some(config))
      transform
    }

    describe("save & load"){
      it("should load a Word2VecTransformer"){
        val transform = loadModel
        transform.vectors shouldBe a [Map[_, Array[Float]]]
        transform.vectors.size shouldBe 433
        transform.vectorSize shouldBe 100
      }

      it("should save a Word2VecTransformer"){
        val transform = loadModel
        val json = transform.save(intentAlloy.writer)
        implicit val formats = DefaultFormats
        val outputPath = (json.get \ "path").extract[String]
        outputPath shouldBe path
      }
    }

    describe("apply") {
      val transform = loadModel

      it("should return an empty sparse vector of size 100 on empty input") {
        val expected = Vectors.sparse(100, Array.empty[Int], Array.empty[Double])
        transform(Seq[String]()) shouldBe expected
      }

      it("should return a vector of non-zero floats with a single word input") {
        val oneWordSeq = Seq[String]("anarchist")
        val outputVector = transform(oneWordSeq)
        outputVector should have size 100
        for (elem <- outputVector.toArray)
          elem should (be > -1.0 and be < 1.0 and not be 0)
      }

      it("should return a vector of non-zero floats with a multi-word input") {
        val multiwordSeq = Seq[String]("kropotkin", "is", "considered", "an", "anarchist")
        val outputVector = transform(multiwordSeq).toArray
        outputVector should have size 100
        for (elem <- outputVector.toArray)
          elem should (be > -1.0 and be < 1.0 and not be 0)
      }

      it("should raise an exception on out-of-vocabulary input") {
        val oneWordSeq = Seq[String]("batman")
        val thrown = intercept[IllegalStateException] {
          transform(oneWordSeq).toArray
        }
        assert(thrown.getMessage === "batman not in vocabulary")
      }
    }
  }
}