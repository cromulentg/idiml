package com.idibon.ml.feature.tokenizer

import org.scalatest.{Matchers, FunSpec}
import com.ibm.icu.text.{UForwardCharacterIterator, UCharacterIterator}

class URIStateMachineSpec extends FunSpec with Matchers {

  def uriLengthOf(x: String, tpe: URIStateMachineState.Value): Int = {
    val it = UCharacterIterator.getInstance(x)
    val sm = new URIStateMachine().reset(tpe)
    var last = UForwardCharacterIterator.DONE
    Stream.continually({last = it.nextCodePoint; last})
      .takeWhile(cp => sm.nextCodePoint(cp).hasNext)
      .force
    if (last != UForwardCharacterIterator.DONE) it.previousCodePoint
    it.getIndex
  }

  def uriHierLengthOf(x: String) = uriLengthOf(x, URIStateMachineState.AUTHORITY)
  def uriOpaqLengthOf(x: String) = uriLengthOf(x, URIStateMachineState.OPAQUE)

  it("should return 0 for empty strings") {
    uriHierLengthOf("") shouldBe 0
    uriOpaqLengthOf("") shouldBe 0
  }

  it("should stop on whitespace and control characters") {
    uriHierLengthOf(" ") shouldBe 0
    uriOpaqLengthOf("\t") shouldBe 0
    uriOpaqLengthOf("\u000b") shouldBe 0
    uriHierLengthOf("\u205f") shouldBe 1  // this whitespace character is allowed
  }

  it("should support surrogate pairs in hierarchical IRIs") {
    uriHierLengthOf("www.\ud83d\udca9.com/ ") shouldBe 11
    uriOpaqLengthOf("www.\ud83d\udca9.com/ ") shouldBe 4
  }

  it("should terminate on expected boundary characters") {
    uriHierLengthOf("www.google.com?q=1#!foo#foo") shouldBe 23
    uriOpaqLengthOf("/test") shouldBe 0
  }

  it("should support percent-encoding") {
    uriHierLengthOf("www.idibon.com/Test%201") shouldBe 23
    uriHierLengthOf("www.idibon.com/Test%%201") shouldBe 20
  }

  it("should support tel and mailto opaque URIs") {
    uriOpaqLengthOf("employee@idibon.com ") shouldBe 19
    uriOpaqLengthOf("863-1234;phone-context=+1-914-555 ") shouldBe 33
  }
}
