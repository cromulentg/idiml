package com.idibon.ml.alloy

import java.io.{FileOutputStream, DataInputStream, DataOutputStream, File}
import java.util.Random
import java.util.jar.{JarFile, JarOutputStream, Manifest}

import com.idibon.ml.alloy.Alloy
import com.idibon.ml.predict.ensemble.{EnsembleModel, EnsembleModelLoader}
import com.idibon.ml.predict.rules.DocumentRules
import com.idibon.ml.predict.{PredictModel, PredictOptionsBuilder, SingleLabelDocumentResult}
import scala.collection.mutable
import org.json4s._
import org.scalatest._

/**
  * Class to test the JarAlloy.
  *
  * It has a bunch of integration tests that should be fairly stable, unless one
  * of the underlying models changes. Otherwise it contains unit
  */
class ScalaJarAlloySpec extends FunSpec with Matchers with BeforeAndAfter {

  describe("integration tests") {
    it("saves and loads ensemble model of rules as intended") {
      val docRules1 = new DocumentRules("alabel", List())
      val docRules2 = new DocumentRules("alabel", List(("is", 0.5f)))
      val ensemble = new EnsembleModel("alabel", List(docRules1, docRules2))
      val random = new Random().nextLong()
      val filename = s"${random}_test.jar"
      println(filename)
      val labelToModel = new mutable.HashMap[String, PredictModel]()
      labelToModel.put("alabel", ensemble)
      val alloy = new ScalaJarAlloy(labelToModel, new mutable.HashMap[String, String]())
      alloy.save(filename)
      val jarFile = new File(filename)
      jarFile.exists() shouldBe true
      val resurrectedAlloy = ScalaJarAlloy.load(filename)
      val options = new PredictOptionsBuilder().build()
      val documentObject: JsonAST.JObject = JObject(List(("content", JString("content IS awesome"))))
      val result1 = alloy.predict(documentObject, options).get("alabel")
        .asInstanceOf[SingleLabelDocumentResult]
      val result2 = resurrectedAlloy.predict(documentObject, options).get("alabel")
        .asInstanceOf[SingleLabelDocumentResult]
      result2.matchCount shouldBe result1.matchCount
      result2.probability shouldBe result1.probability
      //delete file
      jarFile.delete()
    }
  }
  describe("Tests JarWriter & JarReader") {
    /**
      * helper function to create alloy writer.
      * @return
      */
    def createAlloyWriter(filename: String): (File, JarOutputStream, Alloy.Writer) = {
      val file = new File(filename)
      val jos = new JarOutputStream(new FileOutputStream(file), new Manifest())
      val jw: Alloy.Writer = new JarWriter("", jos)
      (file, jos, jw)
    }
    /**
      * helper function to create alloy reader.
      * @param file
      * @return
      */
    def createAlloyReader(file: File): (JarFile, Alloy.Reader) = {
      val jarFile = new JarFile(file)
      val reader: Alloy.Reader = new JarReader("", jarFile)
      (jarFile, reader)
    }
    it("Can write to and read from the same jar") {
      val (file: File, jos: JarOutputStream, jw: Alloy.Writer) = createAlloyWriter("/tmp/basetest.jar")
      val jw2: Alloy.Writer = jw.within("aFolder")
      var dos: DataOutputStream = jw2.resource("test.json")
      Codec.String.write(dos, "{\"this\": \"is some json!\"}")
      dos.close()
      dos = jw.resource("test2.json")
      Codec.String.write(dos, "{\"this\": \"is some more json!\"}")
      dos.close()
      jos.close()
      val (jarFile: JarFile, reader: Alloy.Reader) = createAlloyReader(file)
      val stream: DataInputStream = reader.resource("test2.json")
      Codec.String.read(stream) shouldEqual "{\"this\": \"is some more json!\"}"
      stream.close()
      val reader2: Alloy.Reader = reader.within("aFolder")
      val stream2: DataInputStream = reader2.resource("test.json")
      Codec.String.read(stream2) shouldEqual "{\"this\": \"is some json!\"}"
      stream2.close()
      jarFile.close()
      //delete file
      file.delete()
    }

    it("We can write CJK resource names") {
      val (file: File, jos: JarOutputStream, jw: Alloy.Writer) = createAlloyWriter("/tmp/cjktest.jar")
      val jw2: Alloy.Writer = jw.within("CJK")
      var dos: DataOutputStream = jw2.resource("日ーご.json")
      Codec.String.write(dos, "{\"これは\": \"ジェゾンです!\"}")
      dos.close()
      dos  = jw2.resource("ABC김기영기자BBC.json")
      Codec.String.write(dos, "{\"ABC김기영기자BBC\": \"ABC김기영기자BBC!\"}")
      dos.close()
      dos = jw2.resource("妈妈李青.json")
      Codec.String.write(dos, "{\"妈妈李青\": \"准生证时被!\"}")
      dos.close()
      jos.close()
      val (jarFile: JarFile, baseReader: Alloy.Reader) = createAlloyReader(file)
      val reader = baseReader.within("CJK")
      var stream: DataInputStream = reader.resource("日ーご.json")
      Codec.String.read(stream) shouldEqual "{\"これは\": \"ジェゾンです!\"}"
      stream.close()
      stream = reader.resource("ABC김기영기자BBC.json")
      Codec.String.read(stream) shouldEqual "{\"ABC김기영기자BBC\": \"ABC김기영기자BBC!\"}"
      stream.close()
      stream = reader.resource("妈妈李青.json")
      Codec.String.read(stream) shouldEqual "{\"妈妈李青\": \"准生证时被!\"}"
      stream.close()
      jarFile.close()
      //delete file
      file.delete()
    }

    it("We can write emoji (unicode) resource names") {
      val (file: File, jos: JarOutputStream, jw: Alloy.Writer) = createAlloyWriter("/tmp/emoji.jar")
      val jw2: Alloy.Writer = jw.within("emoji")
      val dos: DataOutputStream = jw2.resource("\uD83D\uDC35.json")
      Codec.String.write(dos, "{\"\uD83D\uDC26\": \"\uD83D\uDC0D\"}")
      dos.close()
      jos.close()
      val (jarFile: JarFile, baseReader: Alloy.Reader) = createAlloyReader(file)
      val reader = baseReader.within("emoji")
      val stream: DataInputStream = reader.resource("\uD83D\uDC35.json")
      Codec.String.read(stream) shouldEqual "{\"\uD83D\uDC26\": \"\uD83D\uDC0D\"}"
      stream.close()
      jarFile.close()
      //delete file
      file.delete()
    }

    val lyrics: List[String] = """!I went to the moped store, said, "Buck it."
                   |Salesman's like "What up, what's your budget?"
                   |And I'm like "Honestly, I don't know nothing about mopeds."
                   |He said "I got the one for you, follow me."
                   |Oh it's too real
                   |Chromed out mirror, I don't need a windshield
                   |Banana seat, a canopy on two wheels
                   |Eight hundred cash, that's a hell of a deal""".split("\n").map(_.trim()).toList

    it("We can have multiple DataOutputStreams and write to them in a parallel.") {
      val (file: File, jos: JarOutputStream, jw: Alloy.Writer) = createAlloyWriter("/tmp/emoji.jar")
      val jw2: Alloy.Writer = jw.within("parallel")
      lyrics.par //in parallel
        // create resource
        .map(x => (x, jw2.resource(s"${x.charAt(1)}.json")))
        // write to resource
        .map(x => (x._2, Codec.String.write(x._2, x._1)))
        // close
        .foreach(x => x._1.close())
      jos.close()
      val (jarFile: JarFile, baseReader: Alloy.Reader) = createAlloyReader(file)
      val reader = baseReader.within("parallel")
      lyrics.par
        .map(x => (x, reader.resource(s"${x.charAt(1)}.json")))
        .map(x => (x._1, Codec.String.read(x._2), x._2))
        .map(x => (x._1, x._2, x._3.close()))
        .foreach(x => x._2 shouldBe x._1)
      jarFile.close()
      //delete file
      file.delete()
    }
  }

  describe("Test prediction") {
    it("It uses the base class predict method successfully") {
      //TODO: once alloy level predict API is finalized.
    }
  }
}
