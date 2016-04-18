package com.idibon.ml.train.datagenerator

import java.io.{File, FileOutputStream, IOException}
import org.scalatest.{Matchers, FunSpec}

class FileUtilsSpec extends FunSpec with Matchers {

  val rng = new java.util.Random

  describe("createTemporaryDirectory") {
    it("should create unique random path names") {
      val prefix = "fileUtilsSpec"
      var temp1: File = null
      var temp2: File = null

      try {
        temp1 = FileUtils.createTemporaryDirectory(prefix)
        temp1.getName should startWith(prefix)
        temp1.getName should endWith regex("""-\d+""")
        temp1.canWrite shouldBe true
        temp2 = FileUtils.createTemporaryDirectory(prefix)
        temp2.getName should startWith(prefix)
        temp2.getName should endWith regex("""-\d+""")
        temp2.getName should not be temp1.getName
        temp2.canWrite shouldBe true
      } finally {
        if (temp1 != null) temp1.delete
        if (temp2 != null) temp2.delete
      }
    }

    it("should abort with an error if it cant create a name") {
      val r = new java.util.Random {
        // RFC 1149.5 specifies 4 as the standard IEEE-vetted random number
        override def nextInt: Int = 4
      }
      var base: File = null
      try {
        base = FileUtils.createTemporaryDirectory("foo", 1, r)
        base.getName shouldBe "foo-4"
        intercept[IOException] {
          FileUtils.createTemporaryDirectory("foo", 1, r)
        }
      } finally {
        if (base != null) base.delete
      }
    }
  }

  describe("rm_rf") {

    it("should delete single files") {
      val f = File.createTempFile("test", ".txt")
      f.exists shouldBe true
      FileUtils.rm_rf(f)
      f.exists shouldBe false
    }

    it("should delete directories") {
      val root = new File(System.getProperty("java.io.tmpdir"), s"fileUtilsSpec${rng.nextInt(65535)}")
      root.exists shouldBe false
      root.mkdir()
      root.exists shouldBe true
      val os = new FileOutputStream(new File(root, "file"))
      os.close()
      val subdir = new File(root, "subdir")
      subdir.mkdir()
      val os2 = new FileOutputStream(new File(subdir, "file"))
      os2.close()
      FileUtils.rm_rf(root)
      root.exists shouldBe false
    }
  }
}
