define("bundles/content-feedback/models/RatingCollection",["require","exports","module","underscore","./RatingFeedback"],function(require,exports,module){"use strict";function _classCallCheck(t,e){if(!(t instanceof e))throw new TypeError("Cannot call a class as a function")}var t=function(){function defineProperties(o,n){for(var e=0;e<n.length;e++){var t=n[e];t.enumerable=t.enumerable||!1,t.configurable=!0,"value"in t&&(t.writable=!0),Object.defineProperty(o,t.key,t)}}return function(t,e,n){return e&&defineProperties(t.prototype,e),n&&defineProperties(t,n),t}}(),_=require("underscore"),e=require("./RatingFeedback"),n=function(){function RatingCollection(){_classCallCheck(this,RatingCollection),this.models=[],this.page=0,this.totalCount=0}return RatingCollection.prototype.isEmpty=function isEmpty(){return 0===this.models.length},RatingCollection.prototype.setTotalCount=function setTotalCount(t){this.totalCount=t},RatingCollection.prototype.appendPage=function appendPage(t){var n=this;_(t).each(function(t){var o=t.rating,i=t.timestamp,r=t.id,a=new e(o.value,o.active,t.comments.generic,i,r);n.models.push(a)}),this.page+=1},t(RatingCollection,[{key:"length",get:function get(){return this.models.length}},{key:"hasPendingPage",get:function get(){return this.totalCount!==this.length}}]),RatingCollection}();module.exports=n});