package com.idibon.ml.feature.language

import org.scalatest.{BeforeAndAfter, FunSpec, Matchers}

class LanguageCodeSpec extends FunSpec with Matchers with BeforeAndAfter {

  describe("get") {
    it("LanguageCode.get should output the full feature") {
      val languageCodeEng = new LanguageCode(Some("eng"))
      val languageCodeDeu = new LanguageCode(Some("deu"))

      languageCodeEng.get shouldBe LanguageCode(Some("eng"))
      languageCodeDeu.get shouldBe LanguageCode(Some("deu"))
    }

    it("LanguageCode.getAsString should output human-readable strings") {
      val languageCodeEng = new LanguageCode(Some("eng"))
      val languageCodeDeu = new LanguageCode(Some("deu"))

      languageCodeEng.getAsString shouldBe Some("eng")
      languageCodeDeu.getAsString shouldBe Some("deu")
    }
  }
}
