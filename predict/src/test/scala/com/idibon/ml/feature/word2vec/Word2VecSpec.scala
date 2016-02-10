package com.idibon.ml.feature.word2vec

import scala.collection.mutable.HashMap

import com.idibon.ml.alloy.{MemoryAlloyReader, MemoryAlloyWriter}
import com.idibon.ml.common.EmbeddedEngine
import org.apache.spark.mllib.linalg._
import org.json4s._

import org.scalatest._

class Word2VecSpec extends FunSpec with Matchers {

  describe("Word2Vec") {

    val archive = HashMap[String, Array[Byte]]()
    val path = "src/test/resources/fixtures/model"
    val config = JObject(JField("path",JString(path)))

    def loadModel(): Word2VecTransformer = {
      val loader = new Word2VecTransformerLoader()
      val transform = loader.load(
        new EmbeddedEngine, Some(new MemoryAlloyReader(archive.toMap)), Some(config))
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
        val json = transform.save(new MemoryAlloyWriter(archive))
        implicit val formats = DefaultFormats
        val outputPath = (json.get \ "path").extract[String]
        outputPath shouldBe path
      }
    }

    describe("apply") {
      val transform = loadModel
      val zeroVector = Vectors.sparse(100, Array.empty[Int], Array.empty[Double])

      it("should return an empty sparse vector of size 100 on empty input") {
        transform(Seq[String]()) shouldBe zeroVector
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

      it("should return an empty sparse vector on out-of-vocabulary input") {
        val OOVSeq = Seq[String]("batman")
        transform(OOVSeq) shouldBe zeroVector
      }
    }
  }
}
