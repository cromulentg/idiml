package com.idibon.ml.feature.language

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}
import com.idibon.ml.feature.{FeatureInputStream, FeatureOutputStream}
import org.scalatest.{BeforeAndAfter, FunSpec, Matchers}

class LanguageCodeSpec extends FunSpec with Matchers with BeforeAndAfter {

  describe("save / load") {
    it("should save and load language code features") {
      val known = new LanguageCode(Some("eng"))
      val unknown = new LanguageCode(None)
      val os = new ByteArrayOutputStream
      val fos = new FeatureOutputStream(os)
      fos.writeFeature(known)
      fos.writeFeature(unknown)
      val fis = new FeatureInputStream(new ByteArrayInputStream(os.toByteArray))
      fis.readFeature shouldBe known
      fis.readFeature shouldBe unknown
    }
  }

  describe("icuLocale") {
    it("should raise an exception if the language code is invalid") {
      intercept[IllegalArgumentException] {
        LanguageCode(Some("foo"))
      }
    }
  }

  describe("get") {
    it("LanguageCode.get should output the full feature") {
      val languageCodeEng = new LanguageCode(Some("eng"))
      val languageCodeDeu = new LanguageCode(Some("deu"))

      languageCodeEng.get shouldBe LanguageCode(Some("eng"))
      languageCodeDeu.get shouldBe LanguageCode(Some("deu"))
    }

    it("LanguageCode.getHumanReadableString should output human-readable strings") {
      val languageCodeEng = new LanguageCode(Some("eng"))
      val languageCodeDeu = new LanguageCode(Some("deu"))

      languageCodeEng.getHumanReadableString shouldBe Some("eng")
      languageCodeDeu.getHumanReadableString shouldBe Some("deu")
    }
  }
}
