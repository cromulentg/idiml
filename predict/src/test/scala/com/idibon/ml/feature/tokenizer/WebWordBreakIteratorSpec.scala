package com.idibon.ml.feature.tokenizer

import java.lang.ThreadLocal

import com.ibm.icu
import org.scalatest.{Matchers, FunSpec}

class WebWordBreakIteratorSpec extends FunSpec with Matchers {

  val breakIterator = new ThreadLocal[WebWordBreakIterator]() {
    override def initialValue() = new WebWordBreakIterator(
      icu.text.BreakIterator.getWordInstance(icu.util.ULocale.US))
  }

  def !!(x: String): Seq[String] = {
    val i = breakIterator.get
    i.setText(x)
    val boundaries = (i.first ::
      Stream.continually(i.next)
      .takeWhile(_ != icu.text.BreakIterator.DONE).toList)
    boundaries.sliding(2).map(tok => x.substring(tok.head, tok.last)).toSeq
  }

  def tag(x: String): Seq[(String, Int)] = {
    val i = breakIterator.get
    i.setText(x)
    val boundaries = ((i.first, 0) ::
      Stream.continually((i.next, i.getRuleStatus))
      .takeWhile(_._1 != icu.text.BreakIterator.DONE).toList)
    boundaries.sliding(2).map(tok => {
      (x.substring(tok.head._1, tok.last._1), tok.last._2)
    }).toSeq
  }

  it("should support empty strings") {
    !!("") shouldBe List("")
  }

  it("should report a negative status tag for custom tokens") {
    tag("Our website is http://www.idibon.com. &quot;:)") shouldBe List(
      ("Our", 200), (" ", 0), ("website", 200), (" ", 0), ("is", 200),
      (" ", 0), ("http://www.idibon.com", Tag.ruleStatus(Tag.URI)),
      (".", 0), (" ", 0), ("&quot;", Tag.ruleStatus(Tag.Word)),
      (":)", Tag.ruleStatus(Tag.Word)))
  }

  it("should tokenize character references") {
    !!("&lt;&gt;&#x20;&#123;") shouldBe List("&lt;", "&gt;", "&#x20;", "&#123;")
    !!("&lt;a href=\"http://www.idibon.com/index.html\"&gt;") shouldBe List(
      "&lt;", "a", " ", "href", "=", "\"", "http://www.idibon.com/index.html",
      "\"", "&gt;")
  }

  it("should not tokenize invalid character references") {
    !!("&lt") shouldBe List("&", "lt")
    !!("&#123") shouldBe List("&", "#", "123")
    !!("&#xf00g;") shouldBe List("&", "#", "xf00g", ";")
    !!("&#x110000;") shouldBe List("&", "#", "x110000", ";")
  }

  ignore("should split words that have internal colons") {
    !!(":bar") shouldBe List(":", "bar")
    !!(";bar") shouldBe List(";", "bar")
    !!("foo:bar") shouldBe List("foo", ":", "bar")
    !!("foo;bar") shouldBe List("foo", ";", "bar")
  }

  it("should ignore trailing punctuation on URIs") {
    !!("hello: http://www.hamsterdance.org/hamsterdance/!!") shouldBe List(
      "hello", ":", " ", "http://www.hamsterdance.org/hamsterdance/", "!", "!")
    !!("http://www.hamsterdance.org/hamsterdance/. http://www.hamsterdance.org/hamsterdance\ud800\udd02") shouldBe List(
      "http://www.hamsterdance.org/hamsterdance/", ".", " ",
      "http://www.hamsterdance.org/hamsterdance", "\ud800\udd02")
  }

  it("should not tokenize URIs when there is no URI match") {
    !!("http://%foo") shouldBe List("http", ":/", "/", "%", "foo")
    !!("file://") shouldBe List("file", ":/", "/")
  }

  it("should group combining marks with trailing punctuation") {
    !!("http://www.idibon.com!\u20DD") shouldBe List(
      "http://www.idibon.com", "!\u20DD")
    !!("http://www.idibon.com/\u20e2...") shouldBe List(
      "http://www.idibon.com/\u20e2", ".", ".", ".")
  }

  it("should support mailto: and tel: URIs") {
    !!("(mailto:employee@idibon.com):)") shouldBe List(
      "(", "mailto:employee@idibon.com", ")", ":)")
    !!("Call me at: tel:7042;phone-context=example.com") shouldBe List(
      "Call", " ", "me", " ", "at", ":", " ", "tel:7042;phone-context=example.com")
  }
}
