package com.idibon.ml.alloy

import java.io._
import java.util.Random
import java.util.jar.{JarFile, JarOutputStream, Manifest}

import com.idibon.ml.predict.ensemble.{EnsembleModel}
import com.idibon.ml.predict.rules.DocumentRules
import com.idibon.ml.predict.{EmbeddedEngine, PredictModel, PredictOptionsBuilder, SingleLabelDocumentResult}
import scala.collection.mutable
import org.json4s._
import org.scalatest._

/**
  * Class to test the JarAlloy.
  *
  * It has a bunch of integration tests that should be fairly stable, unless one
  * of the underlying models changes. Otherwise it contains unit tests for JarReader &
  * JarWriter.
  *
  * See http://doc.scalatest.org/2.2.6/#org.scalatest.BeforeAndAfter about before & after and
  * why it's okay to do what is going on here with the temp file deletion.
  *
  * Specifically this block:
  * <quote>
  *  Note that the only way before and after code can communicate with test code is via some
  *  side-effecting mechanism, commonly by reassigning instance vars or by changing the state
  *  of mutable objects held from instance vals (as in this example). If using instance vars
  *  or mutable objects held from instance vals you wouldn't be able to run tests in parallel
  *  in the same instance of the test class unless you synchronized access to the shared,
  *  mutable state. This is why ScalaTest's ParallelTestExecution trait extends
  *  OneInstancePerTest. By running each test in its own instance of the class, each test has
  *  its own copy of the instance variables, so you don't need to synchronize. Were you to mix
  *  ParallelTestExecution into the ExampleSuite above, the tests would run in parallel just
  *  fine without any synchronization needed on the mutable StringBuilder and
  *  ListBuffer[String] objects.
  * </quote>
  */
class ScalaJarAlloySpec extends FunSpec with Matchers with BeforeAndAfter with ParallelTestExecution {

  var tempFilename = ""
  before {
    tempFilename = ""
  }

  after {
    try {
      new File(tempFilename).delete()
    } catch {
      case ioe: IOException => None
    }
  }

  describe("integration tests") {
    it("saves and loads ensemble model of rules as intended") {
      val docRules1 = new DocumentRules("alabel", List())
      val docRules2 = new DocumentRules("alabel", List(("is", 0.5f)))
      val ensemble = new EnsembleModel("alabel", List(docRules1, docRules2))
      val random = new Random().nextLong()
      val labelToModel = new mutable.HashMap[String, PredictModel]()
      labelToModel.put("alabel", ensemble)
      val alloy = new ScalaJarAlloy(labelToModel, new mutable.HashMap[String, String]())
      tempFilename = s"test_${random}.jar"
      alloy.save(tempFilename)
      // let's make sure to delete the file on exit
      val jarFile = new File(tempFilename)
      jarFile.exists() shouldBe true
      // get alloy back & predict on it.
      val resurrectedAlloy = ScalaJarAlloy.load(new EmbeddedEngine, tempFilename)
      val options = new PredictOptionsBuilder().build()
      val documentObject: JsonAST.JObject = JObject(List(("content", JString("content IS awesome"))))
      val result1 = alloy.predict(documentObject, options).get("alabel")
        .asInstanceOf[SingleLabelDocumentResult]
      val result2 = resurrectedAlloy.predict(documentObject, options).get("alabel")
        .asInstanceOf[SingleLabelDocumentResult]
      result2.matchCount shouldBe result1.matchCount
      result2.probability shouldBe result1.probability
    }
  }

  describe("Loads as intended") {
    it("Throws exception on unhandled version") {
      intercept[IOException] {
        val jar = new File(".", "/predict/src/test/resources/fixtures/invalid_alloy_version.jar")
        ScalaJarAlloy.load(null, jar.getCanonicalPath())
      }
    }
  }

  describe("Tests JarWriter & JarReader") {
    /**
      * Helper function to create alloy writer.
      * @return
      */
    def createAlloyWriter(filename: String): (File, JarOutputStream, Alloy.Writer) = {
      val file = File.createTempFile(filename, null)
      tempFilename = filename
      val jos = new JarOutputStream(new FileOutputStream(file), new Manifest())
      val jw: Alloy.Writer = new JarWriter("", jos)
      (file, jos, jw)
    }
    /**
      * Helper function to create alloy reader.
      * @param file
      * @return
      */
    def createAlloyReader(file: File): (JarFile, Alloy.Reader) = {
      val jarFile = new JarFile(file)
      val reader: Alloy.Reader = new JarReader("", jarFile)
      (jarFile, reader)
    }
    /**
      * Helper function to write to a resource.
      * @param writer
      * @param resource
      * @param value
      */
    def writeToResource(writer: Alloy.Writer, resource: String, value: String): Unit = {
      val dos: DataOutputStream = writer.resource(resource)
      Codec.String.write(dos, value)
      dos.close()
    }
    /**
      * Helper function to read from a resource and return the value.
      * @param reader
      * @param resource
      * @return
      */
    def readFromResource(reader: Alloy.Reader, resource: String): String = {
      val stream: DataInputStream = reader.resource(resource)
      val value = Codec.String.read(stream)
      stream.close()
      value
    }

    it("Can write to and read from the same jar") {
      val (file: File, jos: JarOutputStream, jw: Alloy.Writer) = createAlloyWriter("basetest.jar")
      val jw2: Alloy.Writer = jw.within("aFolder")
      writeToResource(jw2, "test.json", "{\"this\": \"is some json!\"}")
      writeToResource(jw, "test2.json", "{\"this\": \"is some more json!\"}")
      jos.close()
      val (jarFile: JarFile, reader: Alloy.Reader) = createAlloyReader(file)
      readFromResource(reader, "test2.json") shouldEqual "{\"this\": \"is some more json!\"}"
      val reader2: Alloy.Reader = reader.within("aFolder")
      readFromResource(reader2, "test.json") shouldEqual "{\"this\": \"is some json!\"}"
      jarFile.close()
    }

    it("We can write CJK resource names") {
      val (file: File, jos: JarOutputStream, jw: Alloy.Writer) = createAlloyWriter("cjktest.jar")
      val jw2: Alloy.Writer = jw.within("CJK")
      val dataPairs: List[(String, String)] = List(
        ("日ーご.json", "{\"これは\": \"ジェゾンです!\"}"),
        ("ABC김기영기자BBC.json", "{\"ABC김기영기자BBC\": \"ABC김기영기자BBC!\"}"),
        ("妈妈李青.json", "{\"妈妈李青\": \"准生证时被!\"}"))
      dataPairs.foreach(x => writeToResource(jw2, x._1, x._2))
      jos.close()
      val (jarFile: JarFile, baseReader: Alloy.Reader) = createAlloyReader(file)
      val reader = baseReader.within("CJK")
      dataPairs.foreach(x => readFromResource(reader, x._1) shouldEqual x._2)
      jarFile.close()
    }

    it("We can write emoji (unicode) resource names") {
      val (file: File, jos: JarOutputStream, jw: Alloy.Writer) = createAlloyWriter("emoji.jar")
      val jw2: Alloy.Writer = jw.within("emoji")
      writeToResource(jw2, "\uD83D\uDC35.json", "{\"\uD83D\uDC26\": \"\uD83D\uDC0D\"}")
      jos.close()
      val (jarFile: JarFile, baseReader: Alloy.Reader) = createAlloyReader(file)
      val reader = baseReader.within("emoji")
      readFromResource(reader, "\uD83D\uDC35.json") shouldEqual "{\"\uD83D\uDC26\": \"\uD83D\uDC0D\"}"
      jarFile.close()
    }

    val lyrics: List[String] =
      """!I went to the moped store, said, "Buck it."
        |Salesman's like "What up, what's your budget?"
        |And I'm like "Honestly, I don't know nothing about mopeds."
        |He said "I got the one for you, follow me."
        |Oh it's too real
        |Chromed out mirror, I don't need a windshield
        |Banana seat, a canopy on two wheels
        |Eight hundred cash, that's a hell of a deal""".split("\n").map(_.trim()).toList

    it("We can have multiple DataOutputStreams and write to them in a parallel.") {
      val (file: File, jos: JarOutputStream, jw: Alloy.Writer) = createAlloyWriter("parallel.jar")
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
    }
  }

  describe("Test prediction") {
    it("It uses the base class predict method successfully") {
      //TODO: once alloy level predict API is finalized.
    }
  }
}
