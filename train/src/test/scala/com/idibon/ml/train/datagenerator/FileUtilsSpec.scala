package com.idibon.ml.train.datagenerator

import java.io.{File, FileOutputStream}
import org.scalatest.{Matchers, FunSpec}

class FileUtilsSpec extends FunSpec with Matchers {

  val rng = new java.util.Random

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
