/* Keep this feature order and split comparison identical to src/models.py. */
(function (root) {
  function treeValue(node, row) {
    while (!Object.prototype.hasOwnProperty.call(node, 'value')) {
      node = row[node.feature] <= node.threshold ? node.left : node.right;
    }
    return node.value;
  }
  function predict(model, row) {
    if (!row.length || row.some(x => !Number.isFinite(x))) throw new Error('Ugyldig måling');
    if (model.kind === 'constant') return model.value;
    if (model.kind === 'ridge') return row.reduce((v, x, i) =>
      v + (x - model.mean[i]) / model.scale[i] * model.weights[i + 1], model.weights[0]);
    const trees = model.kind === 'tree' ? [model.tree] : model.trees;
    return trees.reduce((sum, tree) => sum + treeValue(tree, row), 0) / trees.length;
  }
  root.Vinlab = {predict, treeValue};
  if (typeof module !== 'undefined') module.exports = root.Vinlab;
})(typeof globalThis === 'undefined' ? window : globalThis);
