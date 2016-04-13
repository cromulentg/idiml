package com.idibon.ml.feature.stopwords

import com.idibon.ml.alloy.{Codec, MemoryAlloyReader, MemoryAlloyWriter}
import com.idibon.ml.common.EmbeddedEngine
import com.idibon.ml.feature.language.LanguageCode
import org.json4s.JsonAST.{JString, JObject}
import org.scalatest.{Matchers, FunSpec}
import com.idibon.ml.feature.bagofwords.Word

import scala.collection.mutable

/**
  * Created by nick on 3/21/16.
  */
class StopwordsTransformerSpec extends FunSpec with Matchers {

  val transform = new StopwordsTransformerLoader().load(null, None, None)

  describe("Should work on an empty sequence") {
    it("should work on an empty sequence") {
      transform(Seq[Word](), LanguageCode((Some("eng")))) shouldBe empty
    }
  }

  describe("the cat in the hat --> cat, hat") {
    it("should remove 'The', 'in', and 'the'") {
      val sentence = List(
        Word("the"),
        Word("cat"),
        Word("in"),
        Word("the"),
        Word("hat")
      )
      val expected = List(new Word("cat"), new Word("hat"))
      transform(sentence, LanguageCode(Some("eng"))) shouldBe expected
    }
  }

  describe("questo è un segno diacritico --> segno diacritico") {
    it("should properly handle the accented e") {
      val sentence = List(
        Word("questo"),
        Word("è"),
        Word("un"),
        Word("segno"),
        Word("diacritico")
      )
      val expected = List(new Word("segno"), new Word("diacritico"))
      transform(sentence, LanguageCode(Some("ita"))) shouldBe expected
    }

    it("should properly handle decomposed characters") {
      val sentence = List(
        Word("questo"),
        Word("\u0065\u0301"),
        Word("un"),
        Word("segno"),
        Word("diacritico")
      )
      val expected = List(new Word("segno"), new Word("diacritico"))
      transform(sentence, LanguageCode(Some("ita"))) shouldBe expected
    }
  }

  describe("il nome della rosa --> nome, rosa") {
    it("should remove Il, della") {
      val sentence = List(
        Word("il"),
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
        Word("the"),
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
        Word("the"),
        Word("cat"),
        Word("in"),
        Word("the"),
        Word("hat")
      )
      val expected = sentence
      transform(sentence, LanguageCode(None)) shouldBe expected
    }
  }

  describe("save & load tests") {
    it("saves and loads defaults") {
      val stopwords = Map(
        "ita" -> Set("IL", "UN", "della", "il", "un", "questo", "QUESTO", "È", "DELLA", "è"),
        "eng" -> Set("IN", "in", "THE", "A", "a", "AND", "and", "the")
      )
      val transform = new StopwordsTransformer(stopwords)
      // Save the results
      val archive = mutable.HashMap[String, Array[Byte]]()
      transform.save(new MemoryAlloyWriter(archive))

      val transform2 = (new StopwordsTransformerLoader).load(
        new EmbeddedEngine, Some(new MemoryAlloyReader(archive.toMap)), None)

      transform2.stopwordMap shouldBe transform.stopwordMap
      transform2.defaultStopwordLanguage shouldBe transform.defaultStopwordLanguage
    }
    it("saves and loads custom entries") {
      val stopwords = Map(
        "ita" -> Set("IL", "UN", "della", "il", "un", "questo", "QUESTO", "È", "DELLA", "è"),
        "eng" -> Set("a", "the", "is"),
        "jpn" -> Set("は"))
      val transform = new StopwordsTransformer(stopwords)
      // Save the results
      val archive = mutable.HashMap[String, Array[Byte]]()
      transform.save(new MemoryAlloyWriter(archive))

      val transform2 = (new StopwordsTransformerLoader).load(
        new EmbeddedEngine, Some(new MemoryAlloyReader(archive.toMap)), None)

      transform2.stopwordMap shouldBe transform.stopwordMap
      transform2.defaultStopwordLanguage shouldBe transform.defaultStopwordLanguage
    }

    it("loads custom entries from config that override defaults & saving and loading works") {
      val resourceFile = getClass.getClassLoader.getResource("fixtures/extra_stopwords.txt")
      val config = JObject(List(
        ("languages", JObject(List(
          ("eng", JString(s"file://${resourceFile.getFile()}")),
          ("jpn", JString(s"file://${resourceFile.getFile()}"))
        ))),
        ("defaultStopwordLanguage", JString("eng"))
      ))
      val transform = (new StopwordsTransformerLoader).load(
        new EmbeddedEngine, None, Some(config))
      transform.stopwordMap shouldBe Map(
        "eng" -> Set("hello", "this", "is", "a", "test"),
        "ita" -> Set("É", "IL", "UN", "è", "é", "É", "della", "é", "il", "un", "È",
          "questo", "QUESTO", "È", "DELLA", "è"),
        "jpn" -> Set("hello", "this", "is", "a", "test")
      )
      transform.defaultStopwordLanguage shouldBe "eng"
      val archive = mutable.HashMap[String, Array[Byte]]()
      transform.save(new MemoryAlloyWriter(archive))

      val transform2 = (new StopwordsTransformerLoader).load(
        new EmbeddedEngine, Some(new MemoryAlloyReader(archive.toMap)), None)

      transform2.stopwordMap shouldBe transform.stopwordMap
      transform2.defaultStopwordLanguage shouldBe transform.defaultStopwordLanguage
    }
    it("saves & loads custom default stopword language") {
      val stopwords = Map(
        "ita" -> Set("IL", "UN", "della", "il", "un", "questo", "QUESTO", "È", "DELLA", "è"),
        "eng" -> Set("IN", "in", "THE", "A", "a", "AND", "and", "the")
      )
      val transform = new StopwordsTransformer(stopwords, "eng")
      // Save the results
      val archive = mutable.HashMap[String, Array[Byte]]()
      transform.save(new MemoryAlloyWriter(archive))

      val transform2 = (new StopwordsTransformerLoader).load(
        new EmbeddedEngine, Some(new MemoryAlloyReader(archive.toMap)), None)

      transform2.stopwordMap shouldBe transform.stopwordMap
      transform2.defaultStopwordLanguage shouldBe transform.defaultStopwordLanguage
    }
    it("loads when no default stopword language was saved (for backwards compatibility)") {
      // Save the results
      val archive = mutable.HashMap[String, Array[Byte]]()
      val writer = new MemoryAlloyWriter(archive)
      val baseStopwords = StopwordsTransformer.generateStopFileMap()
      // write them out -- first what languages
      val baseDir = writer.within(StopwordsTransformer.STOPWORD_RESOURCE_NAME)
      val languages = baseDir.resource(StopwordsTransformer.STOPWORD_LANGUAGE_RESOURCE_NAME)
      Codec.VLuint.write(languages, baseStopwords.size)
      baseStopwords.foreach({case (code, _) =>
        Codec.String.write(languages, code)
      })
      languages.close()
      // second the actual words into each language resource
      baseStopwords.foreach({case (code, words) =>
        val resource = baseDir.resource(code)
        Codec.VLuint.write(resource, words.size)
        words.foreach(w => Codec.String.write(resource, w))
        resource.close()
      })
      val transform2 = (new StopwordsTransformerLoader).load(
        new EmbeddedEngine, Some(new MemoryAlloyReader(archive.toMap)), None)
      val transform1 = new StopwordsTransformer(baseStopwords, "")
      transform2.stopwordMap shouldBe transform1.stopwordMap
      transform2.defaultStopwordLanguage shouldBe transform1.defaultStopwordLanguage
    }

  }
}



