package com.idibon.ml.alloy
import org.scalatest._
import java.io.{ByteArrayOutputStream, ByteArrayInputStream,
  EOFException, IOException}

class CodecSpec extends FunSpec with Matchers {

  describe("VLuint") {

    it("should encode values compactly") {
      val stream = new ByteArrayOutputStream
      Codec.VLuint.write(stream, 0)
      Codec.VLuint.write(stream, 16)
      Codec.VLuint.write(stream, 64)
      Codec.VLuint.write(stream, 127)
      Codec.VLuint.write(stream, 128)
      Codec.VLuint.write(stream, 255)
      Codec.VLuint.write(stream, 4096)
      Codec.VLuint.write(stream, 16383)
      Codec.VLuint.write(stream, 16384)
      Codec.VLuint.write(stream, (1 << 20) - 1)
      Codec.VLuint.write(stream, (1 << 20) + 1)
      stream.toByteArray shouldEqual Array[Byte](0, 16, 64, 127,
        -128, 1, -1, 1, -128, 32, -1, 127, -128, -128, 1,
        -1, -1, 63, -127, -128, 64)
    }

    it("should raise an exception on negative values") {
      val stream = new ByteArrayOutputStream
      intercept[IllegalArgumentException] {
        Codec.VLuint.write(stream, -1)
      }
    }

    it("should decode encoded values") {
      val stream = new ByteArrayInputStream(Array[Byte](0, 16,
        64, 127, -128, 1, -1, 1, -128, 32, -1, 127, -128,
        -128, 1, -1, -1, 63, -127, -128, 64, 0, 127))
      Codec.VLuint.read(stream) shouldBe 0
      Codec.VLuint.read(stream) shouldBe 16
      Codec.VLuint.read(stream) shouldBe 64
      Codec.VLuint.read(stream) shouldBe 127
      Codec.VLuint.read(stream) shouldBe 128
      Codec.VLuint.read(stream) shouldBe 255
      Codec.VLuint.read(stream) shouldBe 4096
      Codec.VLuint.read(stream) shouldBe 16383
      Codec.VLuint.read(stream) shouldBe 16384
      Codec.VLuint.read(stream) shouldBe ((1 << 20) - 1)
      Codec.VLuint.read(stream) shouldBe ((1 << 20) + 1)
      Codec.VLuint.read(stream) shouldBe 0
      Codec.VLuint.read(stream) shouldBe 127
    }

    it("should raise an exception on incomplete integers") {
      val stream = new ByteArrayInputStream(Array[Byte](-128))
      intercept[EOFException] {
        Codec.VLuint.read(stream)
      }
    }

    it("should raise an exception if the encoded value is too big") {
      val stream = new ByteArrayInputStream(Array[Byte](-128, -128,
        -128, -128, -128))
      intercept[IOException] {
        Codec.VLuint.read(stream)
      }
    }
  }

  describe("String") {

    it("should encode strings in real UTF-8") {
      val stream = new ByteArrayOutputStream
      /* Java uses a "modified" UTF-8 format in DataOutputStream#writeUTF
       * which encodes emoji as the UTF-8 encoded bytes of surrogate pairs,
       * rather than the UTF-8 encoding of the actual codepoint. make sure
       * that our encoding is *real* UTF-8 */
      Codec.String.write(stream, "\ud83d\udca9")
      Codec.String.write(stream, "Dance: \ud83d\udc83")
      stream.toByteArray shouldEqual Array[Byte](4, -16, -97, -110, -87,
        11, 68, 97, 110, 99, 101, 58, 32, -16, -97, -110, -125)
    }

    it("should decode strings") {
      val stream = new ByteArrayInputStream(Array[Byte](4, -16, -97,
        -110, -87, 11, 68, 97, 110, 99, 101, 58, 32, -16, -97, -110, -125))
      Codec.String.read(stream) shouldBe "\ud83d\udca9"
      Codec.String.read(stream) shouldBe "Dance: \ud83d\udc83"
    }

    it("should encode 0-length strings") {
      val stream = new ByteArrayOutputStream
      Codec.String.write(stream, "")
      stream.toByteArray shouldEqual Array[Byte](0)
    }

    it("should read 0-length strings") {
      val stream = new ByteArrayInputStream(Array[Byte](0))
      Codec.String.read(stream) shouldBe ""
    }

    it("should raise an exception on truncated strings") {
      val stream = new ByteArrayInputStream(Array[Byte](4, -16, -97))
      intercept[EOFException] {
        Codec.String.read(stream)
      }
    }

    it("should raise an exception if the bytestream is not UTF-8") {
      val notUtf8 = "Ümlaut".getBytes("ISO-8859-1")
      val generator = new ByteArrayOutputStream
      Codec.VLuint.write(generator, notUtf8.length)
      generator.write(notUtf8)
      val stream = new ByteArrayInputStream(generator.toByteArray)
      intercept[java.nio.charset.CharacterCodingException] {
        Codec.String.read(stream)
      }
    }
  }
}
