package com.idibon.ml.predict.crf

import java.io._
import scala.util.Random

import com.idibon.ml.feature._
import com.idibon.ml.predict._

import org.apache.spark.mllib.linalg.Vectors

import org.scalatest.{Matchers, FunSpec}

class FactorieCRFSpec extends FunSpec with Matchers {

  describe("observe") {
    it("should generate sequences") {
      val model = new FactorieCRF(2) with TrainableFactorieModel
      val obs = model.observe(Seq("O" -> Vectors.dense(0.0, 1.0),
        "B0" -> Vectors.dense(1.0, 0.0), "I0" -> Vectors.dense(1.0, 0.0)))
      obs.head.hasPrev shouldBe false
      obs.head.hasNext shouldBe true
      obs.head.target.value.category shouldBe "O"
      obs.head.next.hasPrev shouldBe true
      obs.head.next.next.hasNext shouldBe false
    }
  }

  describe("versioning") {
    it("should throw an exception when loading the wrong version") {
      val bytes = Array[Byte](0)
      intercept[UnsupportedOperationException] {
        FactorieCRF.deserialize(new ByteArrayInputStream(bytes))
      }
    }
  }

  describe("train") {
    def $(label: String, v0: Double, vec: Double*) = (label -> Vectors.dense(v0, vec: _*))

    it("should learn parameters") {
      val documents = Seq(
        Seq($("O", 0.0, 0.0, 0.0, 0.0, 1.0),
          $("B", 0.0, 0.0, 0.0, 1.0, 0.0),
          $("I", 0.0, 0.0, 1.0, 0.0, 0.0),
          $("I", 0.0, 0.0, 0.0, 0.0, 1.0),
          $("I", 0.0, 0.0, 0.0, 0.0, 1.0)),
        Seq($("O", 0.0, 0.0, 0.0, 0.0, 1.0),
          $("O", 0.0, 0.0, 1.0, 0.0, 0.0),
          $("B", 0.0, 0.0, 0.0, 1.0, 0.0),
          $("I", 0.0, 1.0, 1.0, 0.0, 0.0),
          $("O", 1.0, 0.0, 0.0, 0.0, 0.0)))

      val begin = Vectors.dense(0.0, 0.0, 0.0, 1.0, 0.0)
      val outside = Vectors.dense(1.0, 0.0, 0.0, 0.0, 0.0)
      val inside = Vectors.dense(0.0, 0.0, 1.0, 0.0, 0.0)
      val oov = Vectors.zeros(5)

      val model = new FactorieCRF(5) with TrainableFactorieModel
      model.train(documents map model.observe, new Random)

      model.predict(Seq(begin)).head.value.category shouldBe "B"
      model.predict(Seq(outside)).head.value.category shouldBe "O"
      model.predict(Seq(oov)).head.value.category shouldBe "O"
      //model.predict(Seq(inside)).head.value.category shouldBe "O"
      model.predict(Seq(begin, inside)).map(_.value.category) shouldBe Seq("B", "I")

      val expected = documents.reverse.map(doc => {
        model.predict(doc.map(_._2)).map(_.value.category)
      })

      val os = new ByteArrayOutputStream()
      model.serialize(new DataOutputStream(os))
      println(s"Model size: ${os.toByteArray.length}")

      val re = FactorieCRF.deserialize(new ByteArrayInputStream(os.toByteArray))
      re.predict(Seq(begin)).head.value.category shouldBe "B"
      re.predict(Seq(outside)).head.value.category shouldBe "O"

      val compare = documents.reverse.map(doc => {
        re.predict(doc.map(_._2)).map(_.value.category)
      })
      compare shouldBe expected
    }
  }
}
