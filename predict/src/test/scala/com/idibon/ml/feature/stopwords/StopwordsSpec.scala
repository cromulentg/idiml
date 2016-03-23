package com.idibon.ml.feature.stopwords

import com.idibon.ml.feature.language.LanguageCode
import org.scalatest.{Matchers, FunSpec}
import com.idibon.ml.feature.bagofwords.Word

/**
  * Created by nick on 3/21/16.
  */
class StopwordsSpec extends FunSpec with Matchers {

  val transform = new StopwordsTransformer()

  describe("Should work on an empty sequence") {
    it("should work on an empty sequence") {
      transform(Seq[Word](), LanguageCode((Some("eng")))) shouldBe empty
    }
  }

  describe("The cat in the hat --> cat, hat") {
    it("should remove 'The', 'in', and 'the'") {
      val sentence = List(
        Word("The"),
        Word("cat"),
        Word("in"),
        Word("the"),
        Word("hat")
      )
      val expected = List(new Word("cat"), new Word("hat"))
      transform(sentence, LanguageCode(Some("eng"))) shouldBe expected
    }
  }

  describe("Il nome della rosa --> nome, rosa") {
    it("should remove Il, della") {
      val sentence = List(
        Word("Il"),
        Word("nome"),
        Word("della"),
        Word("rosa")
      )
      val expected = List(new Word("nome"), new Word("rosa"))
      transform(sentence, LanguageCode(Some("ita"))) shouldBe expected
    }
  }

  describe("Properly handle unrepresented/bad language code") {
    it("should return the sentence as is") {
      val sentence = List(
        Word("The"),
        Word("cat"),
        Word("in"),
        Word("the"),
        Word("hat")
      )
      val expected = sentence
      transform(sentence, LanguageCode(Some("day"))) shouldBe expected
    }
    it("should return the sentence as is also") {
      val sentence = List(
        Word("The"),
        Word("cat"),
        Word("in"),
        Word("the"),
        Word("hat")
      )
      val expected = sentence
      transform(sentence, LanguageCode(None)) shouldBe expected
    }
  }
}
