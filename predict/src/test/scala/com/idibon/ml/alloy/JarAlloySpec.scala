package com.idibon.ml.alloy

import java.io._
import java.util.jar.{JarFile, JarOutputStream}

import com.idibon.ml.feature.Buildable
import com.idibon.ml.predict._
import com.idibon.ml.predict.ml.TrainingSummary
import com.idibon.ml.predict.ml.metrics._
import scala.collection.mutable.HashMap
import org.scalatest._

class JarAlloySpec extends FunSpec with Matchers {

  describe("Tests JarWriter & JarReader") {
    /**
      * Helper function to create alloy writer.
      *
      * @return
      */
    def createAlloyWriter(filename: String): (File, JarAlloyWriter) = {
      val file = File.createTempFile(filename, null)
      val jos = new JarOutputStream(new FileOutputStream(file))
      (file, new JarAlloyWriter(jos))
    }
    /**
      * Helper function to create alloy reader.
      *
      * @param file
      * @return
      */
    def createAlloyReader(file: File): JarAlloyReader = {
      val jarFile = new JarFile(file)
      new JarAlloyReader(jarFile)
    }
    /**
      * Helper function to write to a resource.
      *
      * @param writer
      * @param resource
      * @param value
      */
    def writeToResource(writer: Alloy.Writer, resource: String, value: String): Unit = {
      val dos: DataOutputStream = writer.resource(resource)
      try {
        Codec.String.write(dos, value)
      } finally {
        dos.close()
      }
    }
    /**
      * Helper function to read from a resource and return the value.
      *
      * @param reader
      * @param resource
      * @return
      */
    def readFromResource(reader: Alloy.Reader, resource: String): String = {
      val stream: DataInputStream = reader.resource(resource)
      try {
        Codec.String.read(stream)
      } finally {
        stream.close
      }
    }

    it("returns null on non-existant resources") {
      val (file: File, jw: JarAlloyWriter) = createAlloyWriter("nulltest.jar")
      var jr: JarAlloyReader = null

      try {
        writeToResource(jw, "resource", "This is a string")
        writeToResource(jw.within("resources"), "resource", "This is also a string")
        jw.jarStream.close
        jr = createAlloyReader(file)
        jr.resource("notAResource") shouldBe null
        readFromResource(jr, "resource") shouldBe "This is a string"
      } finally {
        if (jr != null) jr.jarFile.close
        jw.jarStream.close
        file.delete
      }
    }

    it("Can write to and read from the same jar") {
      val (file: File, jw: JarAlloyWriter) = createAlloyWriter("basetest.jar")
      var reader: JarAlloyReader = null
      try {
        val jw2: Alloy.Writer = jw.within("aFolder")
        writeToResource(jw2, "test.json", "{\"this\": \"is some json!\"}")
        writeToResource(jw, "test2.json", "{\"this\": \"is some more json!\"}")
        jw.jarStream.close()
        reader = createAlloyReader(file)
        readFromResource(reader, "test2.json") shouldEqual "{\"this\": \"is some more json!\"}"
        val reader2: Alloy.Reader = reader.within("aFolder")
        readFromResource(reader2, "test.json") shouldEqual "{\"this\": \"is some json!\"}"
      } finally {
        if (reader != null) reader.jarFile.close
        jw.jarStream.close
        file.delete
      }
    }

    it("We can write CJK resource names") {
      val (file: File, jw: JarAlloyWriter) = createAlloyWriter("cjktest.jar")
      var baseReader: JarAlloyReader = null
      try {
        val jw2: Alloy.Writer = jw.within("CJK")
        val dataPairs: List[(String, String)] = List(
          ("日ーご.json", "{\"これは\": \"ジェゾンです!\"}"),
          ("ABC김기영기자BBC.json", "{\"ABC김기영기자BBC\": \"ABC김기영기자BBC!\"}"),
          ("妈妈李青.json", "{\"妈妈李青\": \"准生证时被!\"}"))
        dataPairs.foreach(x => writeToResource(jw2, x._1, x._2))
        jw.jarStream.close()
        baseReader = createAlloyReader(file)
        val reader = baseReader.within("CJK")
        dataPairs.foreach(x => readFromResource(reader, x._1) shouldEqual x._2)
      } finally {
        if (baseReader != null) baseReader.jarFile.close
        jw.jarStream.close
        file.delete
      }
    }

    it("We can write emoji (unicode) resource names") {
      val (file: File, jw: JarAlloyWriter) = createAlloyWriter("emoji.jar")
      var baseReader: JarAlloyReader = null
      try {
        val jw2: Alloy.Writer = jw.within("emoji")
        writeToResource(jw2, "\uD83D\uDC35.json", "{\"\uD83D\uDC26\": \"\uD83D\uDC0D\"}")
        jw.jarStream.close()
        baseReader = createAlloyReader(file)
        val reader = baseReader.within("emoji")
        readFromResource(reader, "\uD83D\uDC35.json") shouldEqual "{\"\uD83D\uDC26\": \"\uD83D\uDC0D\"}"
      } finally {
        if (baseReader != null) baseReader.jarFile.close
        jw.jarStream.close
        file.delete
      }
    }

    it("returns training summaries when they exist") {
      val alloy = new BaseAlloy("garbage",
        List(new Label("00000000-0000-0000-0000-000000000000", "foo")),
        Map("model for foo" -> new LengthClassificationModel))
        with HasTrainingSummary {
      }
      val archive = HashMap[String, Array[Byte]]()
      val (file: File, jw: JarAlloyWriter) = createAlloyWriter("trainingsummary.jar")
      var baseReader: JarAlloyReader = null
      try {
        alloy.save(jw)
        jw.jarStream.close()
        JarAlloy.getTrainingSummaries(file) shouldBe Seq(new TrainingSummary("testing123",
          Seq[Metric with Buildable[_, _]](
            new FloatMetric(MetricType.AreaUnderROC, MetricClass.Binary, 0.5f),
            new PointsMetric(MetricType.BestF1Threshold, MetricClass.Binary, Seq((0.3f, 0.4f))),
            new LabelIntMetric(MetricType.LabelCount, MetricClass.Binary, "testin123", 23),
            new LabelFloatMetric(MetricType.LabelF1, MetricClass.Binary, "testin123", 0.5f),
            new PropertyMetric(MetricType.HyperparameterProperties, MetricClass.Hyperparameter,
              Seq(("prop1", "value1"))),
            new ConfusionMatrixMetric(MetricType.ConfusionMatrix, MetricClass.Multiclass,
              Seq(("label1", "label2", 2.0f)))
          )))
      } finally {
        jw.jarStream.close()
        file.delete
      }
    }

    it("returns empty sequence when no training summaries exist") {
      val alloy = new BaseAlloy("garbage",
        List(new Label("00000000-0000-0000-0000-000000000000", "foo")),
        Map("model for foo" -> new LengthClassificationModel))
      val archive = HashMap[String, Array[Byte]]()
      val (file: File, jw: JarAlloyWriter) = createAlloyWriter("trainingsummary.jar")
      var baseReader: JarAlloyReader = null
      try {
        alloy.save(jw)
        jw.jarStream.close()
        JarAlloy.getTrainingSummaries(file) shouldBe List()
      } finally {
        jw.jarStream.close()
        file.delete
      }
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
      val (file: File, jw: JarAlloyWriter) = createAlloyWriter("parallel.jar")
      var baseReader: JarAlloyReader = null
      try {
        val jw2: Alloy.Writer = jw.within("parallel")
        lyrics.par //in parallel
           // create resource
          .map(x => (x, jw2.resource(s"${x.charAt(1)}.json")))
          // write to resource
          .map(x => (x._2, Codec.String.write(x._2, x._1)))
          // close
          .foreach(x => x._1.close())
        jw.jarStream.close
        baseReader = createAlloyReader(file)
        val reader = baseReader.within("parallel")
        lyrics.par
          .map(x => (x, reader.resource(s"${x.charAt(1)}.json")))
          .map(x => (x._1, Codec.String.read(x._2), x._2))
          .map(x => (x._1, x._2, x._3.close()))
          .foreach(x => x._2 shouldBe x._1)
      } finally {
        if (baseReader != null) baseReader.jarFile.close
        jw.jarStream.close
        file.delete
      }
    }
  }
}
