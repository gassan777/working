define("bundles/content-feedback/CourseRatingApp",["require","exports","module","bundles/author-branches/reducers/branches","bundles/author-branches/stores/BranchPropertiesStore","bundles/author-branches/reducers/branchSelection","bundles/content-feedback/stores/CourseRatingStore","bundles/teach-course/stores/TeachAppStore","bundles/ssr/stores/ApplicationStore","vendor/cnpm/fluxible.v0-4/index"],function(require,exports,module){"use strict";var r=require("bundles/author-branches/reducers/branches").store,s=require("bundles/author-branches/stores/BranchPropertiesStore"),t=require("bundles/author-branches/reducers/branchSelection").store,o=require("bundles/content-feedback/stores/CourseRatingStore"),n=require("bundles/teach-course/stores/TeachAppStore"),c=require("bundles/ssr/stores/ApplicationStore"),u=require("vendor/cnpm/fluxible.v0-4/index"),e=new u;e.registerStore(r),e.registerStore(s),e.registerStore(t),e.registerStore(n),e.registerStore(o),e.registerStore(c),module.exports=e});