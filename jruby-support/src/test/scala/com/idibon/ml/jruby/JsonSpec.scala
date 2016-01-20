package com.idibon.ml.jruby

import org.jruby.Ruby
import org.jruby.java.proxies.JavaProxy
import org.json4s._
import org.scalatest.{Matchers, FunSpec}

class JsonSpec extends FunSpec with Matchers {

  describe("toIdiml") {

    it("should convert Ruby objects to Json ASTs") {
      val script = """require 'json'
object = JSON.parse('{"foo":"bar","baz":[0, 1, 2]}')
com.idibon.ml.jruby.Json.toIdiml(object)"""
      val result = Ruby.newInstance.evalScriptlet(script)
      result shouldBe a [JavaProxy]
      result.asInstanceOf[JavaProxy].getObject shouldBe a [JObject]
      val json = result.asInstanceOf[JavaProxy].getObject.asInstanceOf[JObject]
      (json \ "foo").asInstanceOf[JString] shouldBe JString("bar")
      (json \ "baz").asInstanceOf[JArray].values should contain theSameElementsInOrderAs List(0, 1, 2)
    }
  }
}
