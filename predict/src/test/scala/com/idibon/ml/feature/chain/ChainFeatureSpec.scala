package com.idibon.ml.feature.chain

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import com.idibon.ml.feature._

import org.scalatest.{Matchers, FunSpec}

class ChainFeatureSpec extends FunSpec with Matchers {

  it("should save and load features at all offsets") {
    (-128 to 127).foreach(offset => {
      val feature = ChainFeature(offset.toByte, StringFeature("foobar"))
      val os = new ByteArrayOutputStream
      val fos = new FeatureOutputStream(os)
      fos.writeFeature(feature)
      val fis = new FeatureInputStream(new ByteArrayInputStream(os.toByteArray))
      val reload = fis.readFeature
      reload shouldBe feature
    })
  }

  it("should upcast sensibly") {
    val f: ChainFeature[Feature[String]] = ChainFeature(0, StringFeature("foobar"))
    f.feature.get shouldBe "foobar"
  }
}
