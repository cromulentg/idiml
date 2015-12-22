package com.idibon.ml.test

import org.apache.logging.log4j.core.Logger
import org.apache.logging.log4j.core.layout.PatternLayout
import org.apache.logging.log4j.core.appender.WriterAppender
import org.apache.logging.log4j.LogManager
import org.apache.logging.log4j.Level

/** Mixin to support verifying log messages */
trait VerifyLogging {
  /** Name of the logger that will be tested
    *
    * Implementations should override with the name of the log4j/slf4j
    * logger that will be used during the unit test, so that tests may
    * spy on the log output.
    */
  val loggerName: String

  private[this] var _logWriter: Option[java.io.StringWriter] = None

  private[this] lazy val _appenderName = VerifyLogging.nextName

  /** Clears out logged messages
    *
    * Clears any accumulated messages written to this object private
    * log. Intended to be called as part of a unit test, or in a
    * per-test before or after block for a test suite.
    */
  def resetLog {
    _logWriter.map(_.getBuffer.setLength(0))
  }

  /** Returns all logged messages
    *
    * Returns all of the messages logged to this object's private log
    * since the last call to resetLog, or an empty string if no private log
    * is configured.
    */
  def loggedMessages = _logWriter.map(_.getBuffer.toString).getOrElse("")

  /** Shuts down the private logging facilities for this object
    *
    * Frees the resources created to support logging introspection for
    * a specific test suite, returning to the Log4j base configuration.
    * Call in the afterAll helper for the test suite.
    */
  def shutdownLogging {
    if (_logWriter.isDefined) {
      val logger = LogManager.getLogger(loggerName).asInstanceOf[Logger]
      logger.removeAppender(logger.getAppenders.get(_appenderName))
      _logWriter = None
    }
  }

  /** Initializes private logging facilities for this object
    *
    * Creates a private logger for loggerName that outputs to a StringWriter
    * so that the logged messages can be inspected for accuracy as part of
    * a unit test suite. Call in the beforeAll helper for the test suite.
    */
  def initializeLogging {
    VerifyLogging.initialize

    val writer = new java.io.StringWriter
    val appender = WriterAppender.createAppender(
      PatternLayout.createDefaultLayout, null, writer,
      _appenderName, false, true)
    appender.start

    val logger = LogManager.getLogger(loggerName).asInstanceOf[Logger]

    logger.addAppender(appender)
    logger.setLevel(Level.INFO)
    _logWriter = Some(writer)
  }
}

/** Global initialization of Log4j infrastructure for unit testing */
private [this] object VerifyLogging {

  /** Generates a unique name for each private appender
    */
  def nextName = s"A${logCount.incrementAndGet}"

  /** Global thread-safe integer ID used to generate unique appender names
    */
  val logCount = new java.util.concurrent.atomic.AtomicInteger(0)

  /** Singleton initialization for the log4j 2 infrastructure
    *
    * Used to configure Log4j without a log4j.properties file in the
    * class-path, which is how it may be run during unit-testing
    */
  lazy val initialize = {
    org.apache.log4j.BasicConfigurator.configure
    true
  }
}
