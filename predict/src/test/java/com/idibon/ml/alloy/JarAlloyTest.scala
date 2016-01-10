package com.idibon.ml.alloy

import java.io.File
import java.util.Random

import com.idibon.ml.predict.ensemble.{EnsembleModelLoader, EnsembleModel}
import com.idibon.ml.predict.rules.DocumentRules
import org.json4s._
import org.scalatest._

/**
  * Created by stefankrawczyk on 1/13/16.
  */
class JarAlloyTest extends FunSpec with Matchers with BeforeAndAfter {

  describe("testReader & writer") {
    it("saves and loads as intended") {
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
      // don't forget to close!
      alloy.close()
      //delete file
      new File(filename).delete()
    }
  }
}
