package com.idibon.ml.feature.word2vec


import scala.collection.mutable.HashMap

import com.idibon.ml.feature.{Feature, StringFeature}
import com.idibon.ml.alloy.{MemoryAlloyReader, MemoryAlloyWriter}
import com.idibon.ml.common.EmbeddedEngine
import org.apache.spark.mllib.linalg._
import org.json4s._

import org.scalatest._

trait Word2VecTransformerBehaviors {this: FunSpec with Matchers =>

  val archive = HashMap[String, Array[Byte]]()

  def loadModel(uri: String, modelType: String): Word2VecTransformer = {
    val config = JObject(JField("uri", JString(uri.toString())), JField("type", JString(modelType)))
    val loader = new Word2VecTransformerLoader()
    val transform = loader.load(
      new EmbeddedEngine, Some(new MemoryAlloyReader(archive.toMap)), Some(config))
    transform
  }

  def aWord2VecTransformer(uri: String, modelType: String, vocabSize: Int) {

    it("should load a Word2VecTransformer with type "+modelType) {
      val transform = loadModel(uri, modelType)
      transform.vectors shouldBe a[Map[_, Array[Float]]]
      transform.vectors.size shouldBe vocabSize
      transform.vectorSize shouldBe 100
    }

    it("should save a Word2VecTransformer with type "+modelType) {
      val transform = loadModel(uri, modelType)
      val json = transform.save(new MemoryAlloyWriter(archive))
      implicit val formats = DefaultFormats
      val outputURI = (json.get \ "uri").extract[String]
      outputURI shouldBe uri
    }


    describe("apply") {
      val transform = loadModel(uri, modelType)
      val zeroVector = Vectors.sparse(100, Array.empty[Int], Array.empty[Double])

      it("should return an empty sparse vector of size 100 on empty input with type "+modelType) {
        transform(Seq[Feature[String]]()) shouldBe zeroVector
      }

      it("should return a vector of non-zero floats with a single word input with type "+modelType) {
        val oneWordSeq = Seq(StringFeature("anarchist"))
        val outputVector = transform(oneWordSeq)
        outputVector should have size 100
        for (elem <- outputVector.toArray)
          elem should (be > -1.0 and be < 1.0 and not be 0)
      }

      it("should return a vector of non-zero floats with a multi-word input with type "+modelType) {
        val multiwordSeq = Seq[Feature[String]](StringFeature("kropotkin"),StringFeature("is"),StringFeature("considered"),StringFeature("an"),StringFeature("anarchist"))
        val outputVector = transform(multiwordSeq).toArray
        outputVector should have size 100
        for (elem <- outputVector.toArray) {
          elem should (be > -1.0 and be < 1.0 and not be 0)
        }
      }

      it("should return an empty sparse vector on out-of-vocabulary input with type "+modelType) {
        val OOVSeq = Seq[Feature[String]](StringFeature("batman"))
        transform(OOVSeq) shouldBe zeroVector
      }
    }
  }
}

class Word2VecSpec extends FunSpec with Matchers with Word2VecTransformerBehaviors {

  describe("Word2VecTransformer") {
    val baseURI = "file://" + System.getProperty("user.dir")
    val sparkModelURI = baseURI + "/src/test/resources/fixtures/sparkWord2VecModel/model"
    val binModelURI = baseURI + "/src/test/resources/fixtures/word2vec-test-vectors.bin.gz"

    it should behave like aWord2VecTransformer(sparkModelURI, "spark", 433)
    it should behave like aWord2VecTransformer(binModelURI, "bin", 434)

  }
}
