package com.idibon.ml.alloy

import java.io.{DataInputStream, DataOutputStream, File}
import java.util.Random

import com.idibon.ml.predict.{SingleLabelDocumentResult, PredictOptionsBuilder}
import com.idibon.ml.predict.ensemble.{EnsembleModelLoader, EnsembleModel}
import com.idibon.ml.predict.rules.DocumentRules
import org.json4s._
import org.scalatest._

/**
  * Class to test the JarAlloy.
  *
  * It has a bunch of integration tests that should be fairly stable, unless one
  * of the underlying models changes. Otherwise it contains unit
  */
class JarAlloySpec extends FunSpec with Matchers with BeforeAndAfter {

  describe("integration tests") {
    it("saves and loads ensemble model of rules as intended") {
      val random = new Random().nextLong()
      val filename = s"${random}_test.jar"
      val alloy = new JarAlloy(filename)
      val docRules1 = new DocumentRules("alabel", List())
      val docRules2 = new DocumentRules("alabel", List(("is", 0.5f)))
      val ensemble = new EnsembleModel("alabel", List(docRules1, docRules2))
      val metadata = ensemble.save(alloy.writer())
      val expectedMetadata = Some(JObject(List(
        ("label",JString("alabel")),
        ("size",JInt(2)),
        ("model-meta",JObject(List(
          ("0", JObject(List(
            ("config", JObject(List(("label",JString("alabel"))))),
            ("class",JString("com.idibon.ml.predict.rules.DocumentRules"))))),
          ("1",JObject(List(
            ("config", JObject(List(("label",JString("alabel"))))),
            ("class",JString("com.idibon.ml.predict.rules.DocumentRules"))))
            )))))))
      metadata shouldBe expectedMetadata
      // don't forget to close!
      alloy.close()
      val ensemble2 = (new EnsembleModelLoader).load(alloy.reader(), metadata)
      ensemble shouldBe ensemble2
      val options = new PredictOptionsBuilder().build()
      val documentObject: JsonAST.JObject = JObject(List(("content", JString("content IS awesome"))))
      val result1 = ensemble.predict(documentObject, options)
        .asInstanceOf[SingleLabelDocumentResult]
      val result2  = ensemble2.predict(documentObject, options)
        .asInstanceOf[SingleLabelDocumentResult]
      result2.matchCount shouldBe result1.matchCount
      result2.probability shouldBe result1.probability

      // don't forget to close!
      alloy.close()
      //delete file
      new File(filename).delete()
    }

    it("Can write to and read from the same jar using alloy constructs.") {
      val filename = "/tmp/asdfsdaf.jar"
      val ja: JarAlloy = new JarAlloy(filename)
      val jw: Alloy.Writer = ja.writer
      val jw2: Alloy.Writer = jw.within("aFolder")
      var dos: DataOutputStream = jw2.resource("test.json")
      Codec.String.write(dos, "{\"this\": \"is some json!\"}")
      dos.close
      dos = jw.resource("test2.json")
      Codec.String.write(dos, "{\"this\": \"is some more json!\"}")
      dos.close
      ja.close
      val reader: Alloy.Reader = ja.reader
      val stream: DataInputStream = reader.resource("test2.json")
      System.out.println(Codec.String.read(stream))
      stream.close
      val reader2: Alloy.Reader = reader.within("aFolder")
      val stream2: DataInputStream = reader2.resource("test.json")
      System.out.println(Codec.String.read(stream2))
      stream2.close
      ja.close
      //delete file
      new File(filename).delete()
    }
  }

  describe("test basic use case") {

  }
}
