package com.idibon.ml.test

import java.util.regex.Pattern
import scala.collection.JavaConversions._
import java.util.Locale
import scala.util.Try
import java.io._

/** Mixin to support verifying log messages */
trait VerifyLogging {
  /** Name of the logger that will be tested
    *
    * Implementations should override with the name of the slf4j
    * logger that will be used during the unit test, so that tests may
    * spy on the log output.
    */
  val loggerName: String

  /** Tracks the location in the recorded stream for this verifier
    *
    * Since there is just a single global recorded byte stream for all
    * logs, we need to track of how many bytes exist in the stream each
    * time "resetLog" is called, to simulate actually clearing the
    * buffer.
    */
  private [this] var _cursor: Option[Int] = None

  /** Clears out logged messages
    *
    * Clears any accumulated messages written to this object private
    * log. Intended to be called as part of a unit test, or in a
    * per-test before or after block for a test suite.
    */
  def resetLog {
    _cursor = _cursor.map(_ => VerifyLogging.size)
  }

  /** Returns all logged messages
    *
    * Returns the first line of each message logged to this object's private
    * log since the last call to resetLog, or an empty string if no private
    * log is configured. The logger name and message level are removed
    */
  def loggedMessages = {
    _cursor.map(VerifyLogging.getRecordedSince(_)
      .split("\\R") // split into lines
      .map(VerifyLogging.LOG_PATTERN.matcher(_)) // only look at log messages
      .filter(m => m.matches && m.group(1) == loggerName) // for this logger
      .map(_.group(2))) // and return the logged message, not the metadata
      .getOrElse(Array[String]()) // or return an empty array
      .mkString("\n")
  }

  /** Shuts down log injection for this test suite
    *
    * Frees the resources created to support logging introspection for
    * the current test suite. Normally it should be called by the
    * test suite's afterAll helper.
    */
  def shutdownLogging {
    VerifyLogging.disable
    _cursor = None
  }

  /** Initializes log injection for this test suite

    * Allocates resources necessary to perform log introspection for
    * a test suite. Normally it should be called by the test suite's
    * beforeAll helper
    */
  def initializeLogging {
    VerifyLogging.enable
    _cursor = Some(VerifyLogging.size)
  }
}

/** Singleton PrintStream used to record slf4j log messages
  *
  * SLF4J-Simple uses a single global PrintStream for all logger
  * instances (meh), which causes all log messages to become
  * intermingled (double-meh). This singleton injects itself with
  * reflection to replace the original TARGET_STREAM, and directs
  * log messages to a byte array output stream created by the
  * object and to the original TARGET_STREAM, so messages continue
  * to appear in the global log.
  *
  * Recording of the log must be enabled / disabled by each test
  * suite that wants to perform log introspection.
  */
private [this] object VerifyLogging
    extends PrintStream(new ByteArrayOutputStream) {

  val LOG_PATTERN = Pattern.compile(
    "(?i)^\\s*(?:WARN|ERROR|INFO|DEBUG|TRACE)\\s+([\\p{L}.$_#\\-0-9:,!@/\\\\]+)\\s+-\\s+(.*)$")

  /** The original TARGET_STREAM created by the SimpleLogger
    *
    * As a side effect of initialization, whatever value previously
    * existed in TARGET_STREAM will be replaced by this object. The
    * entire initialization is wrapped in a critical section to block
    * any log operations from being processed while the initialization
    * happens.
    *
    * Will be Failure if slf4j-simple isn't being used.
    */
  val baseStream: Try[PrintStream] = this.synchronized { Try({
    val klass = Class.forName("org.slf4j.impl.SimpleLogger")

    // create a throw-away instance of SimpleLogger to force initialization
    val ctor = klass.getDeclaredConstructor(classOf[String])
    ctor.setAccessible(true)
    ctor.newInstance("junk")

    val field = klass.getDeclaredField("TARGET_STREAM")
    field.setAccessible(true)
    // preserve the original value
    val original = field.get(null).asInstanceOf[PrintStream]
    field.set(null, this)
    original
  }).recoverWith({ case _ => {
    println("ERROR -- Failed to initialize log injection ('TARGET_STREAM')")
    baseStream
  }}) }

  /** Reference to the "buf" instance variable in ByteArrayOutputStream
    *
    * Used to copy just a subset of the recorded byte stream.
    */
  val fieldBuf: Try[java.lang.reflect.Field] = Try({
    val field = classOf[ByteArrayOutputStream].getDeclaredField("buf")
    field.setAccessible(true)
    field
  }).recoverWith({ case _ => {
    println("ERROR -- Failed to initialize log injection ('buf')")
    fieldBuf
  }})

  // tracks the number of simultaneous clients that want recording
  private[this] var _enableCount = 0

  /** Enables log recording. */
  def enable {
    if (baseStream.isSuccess && fieldBuf.isSuccess) {
      this.synchronized {
        _enableCount += 1
      }
    }
  }

  /** Disables log recording. */
  def disable {
    if (baseStream.isSuccess && fieldBuf.isSuccess) {
      this.synchronized {
        _enableCount -= 1
        // clear any recorded messages if the global state is disabled
        if (_enableCount == 0)
          this.out.asInstanceOf[ByteArrayOutputStream].reset
      }
    }
  }

  /** True when actively recording. */
  def recording = this.synchronized { _enableCount > 0 }

  /** Returns the number of recorded bytes */
  def size = {
    this.synchronized {
      this.out.asInstanceOf[ByteArrayOutputStream].size
    }
  }

  /** Returns the currently-recorded buffer as a String. */
  def getRecordedSince(cursor: Int) = {
    fieldBuf.map(field => new String(this.synchronized {
      val buffer = field.get(this.out).asInstanceOf[Array[Byte]]
      if (cursor < buffer.length)
        java.util.Arrays.copyOfRange(buffer, cursor, buffer.length)
      else
        Array[Byte]()
    })).getOrElse("")
  }

  /* PrintStream layers all of the print* append* and format* methods
   * on top of write, so we only need to override the standard OutputStream
   * methods, rather than overriding all 50 or so print methods */
  override def close {
    this.synchronized {
      super.close
      baseStream.get.close
    }
  }

  override def flush {
    this.synchronized {
      super.flush
      baseStream.get.flush
    }
  }

  override def write(x: Array[Byte], o: Int, l: Int) {
    this.synchronized {
      baseStream.get.write(x, o, l)
      if (recording) super.write(x, o, l)
    }
  }

  override def write(b: Int) {
    this.synchronized {
      baseStream.get.write(b)
      if (recording) super.write(b)
    }
  }
}
