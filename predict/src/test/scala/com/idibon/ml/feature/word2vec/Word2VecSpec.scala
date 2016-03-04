package com.idibon.ml.feature.word2vec


import scala.collection.mutable.HashMap

import com.idibon.ml.feature.{Feature, StringFeature}
import com.idibon.ml.feature.bagofwords.{Word}
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
        transform(Seq[Feature[Word]]()) shouldBe zeroVector
      }

      it("should return a vector of non-zero floats with a single word input with type "+modelType) {
        val oneWordSeq = Seq(Word("anarchist"))
        val outputVector = transform(oneWordSeq)
        outputVector should have size 100
        for (elem <- outputVector.toArray)
          elem should (be > -1.0 and be < 1.0 and not be 0)
      }

      it("should return a vector of non-zero floats with a multi-word input with type "+modelType) {
        val multiwordSeq = Seq[Feature[Word]](Word("kropotkin"),Word("is"),Word("considered"),Word("an"),Word("anarchist"))
        val outputVector = transform(multiwordSeq).toArray
        outputVector should have size 100
        for (elem <- outputVector.toArray) {
          elem should (be > -1.0 and be < 1.0 and not be 0)
        }
      }

      it("should return an empty sparse vector on out-of-vocabulary input with type "+modelType) {
        val OOVSeq = Seq[Feature[Word]](Word("batman"))
        transform(OOVSeq) shouldBe zeroVector
      }
    }
  }
}

class Word2VecSpec extends FunSpec with Matchers with Word2VecTransformerBehaviors {

  val sparkModel = getClass().getClassLoader().getResource("fixtures/sparkWord2VecModel/model")
  val binModel = getClass().getClassLoader().getResource("fixtures/word2vec-test-vectors.bin.gz")

  describe("Word2VecTransformer") {
    it should behave like aWord2VecTransformer(sparkModel.toString(), "spark", 433)
    it should behave like aWord2VecTransformer(binModel.toString(), "bin", 434)

  }

  describe("save / load") {
    it("should be able to load models after saving them") {
      val archive = collection.mutable.HashMap[String, Array[Byte]]()
      val config = loadModel(binModel.toString(), "bin").save(new MemoryAlloyWriter(archive))
      val loader = new Word2VecTransformerLoader()
      val transform = loader.load(new EmbeddedEngine,
        Some(new MemoryAlloyReader(archive.toMap)), config)

      val outputVector = transform(Seq(Word("anarchist")))
      outputVector should have size 100
      for (elem <- outputVector.toArray)
        elem should (be > -1.0 and be < 1.0 and not be 0)
    }
  }
}
