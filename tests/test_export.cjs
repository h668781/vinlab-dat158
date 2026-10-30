/* End-to-end export check against every held-out Python prediction. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.join(__dirname, '..');
const {predict, treeValue} = require('../dist/inference.js');
const artifact = JSON.parse(fs.readFileSync(path.join(root, 'dist/model.json')));
const fixtures = JSON.parse(fs.readFileSync(path.join(root, 'results/prediction_fixtures.json')));
for (const fixture of fixtures) {
  const actual = predict(artifact.model, fixture.input);
  assert.ok(Math.abs(actual - fixture.expected) < 1e-10, `Mismatch: ${actual}, ${fixture.expected}`);
}
assert.throws(() => predict(artifact.model, [NaN]));
assert.throws(() => predict(artifact.model, [Infinity]));
assert.equal(treeValue({feature:0, threshold:2, left:{value:1}, right:{value:5}}, [2]), 1);
console.log(`PASS: ${fixtures.length} Python/JavaScript predictions agree; invalid numbers and split boundary checked.`);
