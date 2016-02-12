package com.idibon.ml.predict

import com.idibon.ml.feature.{FeatureInputStream, FeatureOutputStream, Builder, Buildable}
import java.io.{ByteArrayInputStream, ByteArrayOutputStream}
import org.scalatest.{Matchers, FunSpec}

class LabelSpec extends FunSpec with Matchers {

  describe("save / load") {

    it("should save and load correctly") {
      val bos = new ByteArrayOutputStream
      val l = new Label("baadf00d-0000-5a5a-cdcd-baadbeefcafe", "f00d")
      l.save(new FeatureOutputStream(bos))
      val bis = new ByteArrayInputStream(bos.toByteArray)
      val builder = new LabelBuilder
      builder.build(new FeatureInputStream(bis)) shouldBe l
    }
  }
}
