// Optional Python analysis service.
//
// The application does NOT use this by default, and setting a URL alone does not
// switch it on. Analysis runs in the browser, on the JavaScript port of the estimator
// that tests/regression/test_js_python_agreement.py pins to the Python reference.
//
// Two reasons the remote engine is opt-in rather than automatic:
//   1. The service deployed at this URL exposes a bounded demo endpoint
//      (/api/demo/run, fixed scenarios). It does not implement the general
//      POST /api/analyze contract that the remote engine calls.
//   2. An uploaded CSV is participant data. It stays on the machine unless
//      someone deliberately sends it elsewhere.
//
// To use a service that does implement /api/analyze, set both:
//   window.AEDT_API_URL = "https://...";
//   window.AEDT_ENGINE  = "remote";
window.AEDT_API_URL = "https://aedt-api.onrender.com";
