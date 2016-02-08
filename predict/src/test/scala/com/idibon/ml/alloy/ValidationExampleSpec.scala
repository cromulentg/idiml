package com.idibon.ml.alloy

import java.io.{IOException, File}

import com.idibon.ml.predict.Document
import com.idibon.ml.predict.{PredictResult, ClassificationBuilder, Classification, Document}
import com.idibon.ml.feature.{FeatureInputStream, FeatureOutputStream}
import org.json4s.JsonAST.JObject
import org.scalatest._
import org.json4s._

/**
  * Class to test the ValidationExample.
  *
  */
class ValidationExampleSpec extends FunSpec with Matchers with BeforeAndAfter with ParallelTestExecution {

  describe("it saves & loads"){
    it("saves properly") {
      val c1 = new Classification("test", 0.12f, 1, 0, Seq())
      val v1 = new ValidationExample[Classification](Document.document(JObject(List(JField("nothing", JString("hi"))))), Seq(c1))
      val intentAlloy = new IntentAlloy()
      val out = new FeatureOutputStream(intentAlloy.writer.resource("c1"))
      v1.save(out)
      out.close()
    }
    it("saves and loads properly") {
      val c2 = new Classification("testb", 0.1212f, 1, 0, Seq())
      val v2 = new ValidationExample[Classification](Document.document(JObject(List(JField("nothing", JString("bye"))))), Seq(c2))
      val intentAlloy2 = new IntentAlloy()
      val out = new FeatureOutputStream(intentAlloy2.writer.resource("c2"))
      v2.save(out)
      out.close()
      val b2 = new ValidationExampleBuilder[Classification](new ClassificationBuilder())
      val in = new FeatureInputStream(intentAlloy2.reader.resource("c2"))
      b2.build(in) shouldBe v2
      in.close()
    }
  }
}

/**
  * Class to test the ValidationExamples.
  *
  */
class ValidationExamplesSpec extends FunSpec with Matchers with BeforeAndAfter with ParallelTestExecution {

  describe("it saves & loads"){
    val c1 = new Classification("test", 0.12f, 1, 0, Seq())
    val d1 = Document.document(JObject(List(JField("key1", JString("hi")))))
    val v1 = new ValidationExample[Classification](d1, Seq(c1))

    it("saves properly") {
      val v1s = new ValidationExamples[Classification](List(v1))
      val intentAlloy = new IntentAlloy()
      val out = new FeatureOutputStream(intentAlloy.writer.resource("c1"))
      v1s.save(out)
      out.close()
    }
    it("saves and loads properly") {
      val c2 = new Classification("testb", 0.1212f, 1, 0, Seq())
      val d2 = Document.document(JObject(List(JField("key2", JString("bye")))))
      val v2 = new ValidationExample[Classification](d2, Seq(c2))
      val v2s = new ValidationExamples[Classification](List(v1, v2))
      val intentAlloy2 = new IntentAlloy()
      val out = new FeatureOutputStream(intentAlloy2.writer.resource("c2"))
      v2s.save(out)
      out.close()
      val b2 = new ValidationExamplesBuilder[Classification](new ClassificationBuilder())
      val in = new FeatureInputStream(intentAlloy2.reader.resource("c2"))
      b2.build(in) shouldBe v2s
      in.close()
    }
  }
}
