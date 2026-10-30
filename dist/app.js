const labels = ['Fast syre', 'Flyktig syre', 'Sitronsyre', 'Restsukker', 'Klorid',
  'Fritt svoveldioksid', 'Totalt svoveldioksid', 'Tettleik', 'pH', 'Sulfat', 'Alkohol'];
let artifact;
const el = id => document.getElementById(id);
function clearResult() {
  el('score').textContent = '—'; el('result-label').textContent = 'Målingane er endra';
  el('result-description').textContent = 'Estimer på nytt for å sjå eit oppdatert resultat.';
  el('marker').style.opacity = '0'; el('warning').textContent = '';
}
fetch('model.json').then(response => {
  if (!response.ok) throw new Error('Modellfila kunne ikkje lastast.');
  return response.json();
}).then(data => {
  artifact = data;
  data.features.forEach((feature, i) => {
    const group = document.createElement('div'); group.className = 'field';
    const precision = i === 7 ? 5 : i === 4 ? 3 : 2;
    group.innerHTML = `<label for="f${i}">${labels[i]} <span>${feature.unit}</span></label>
      <input id="f${i}" name="f${i}" type="number" step="any" required min="0" value="${feature.default}">
      <small id="h${i}">Trening: ${feature.min.toFixed(precision)}–${feature.max.toFixed(precision)}</small>`;
    group.querySelector('input').setAttribute('aria-describedby', `h${i}`);
    el('fields').appendChild(group);
  });
  el('mae').textContent = `${data.test.mae.toFixed(3).replace('.', ',')} poeng`;
  el('rows').textContent = data.dataset_rows.toLocaleString('nn-NO');
  el('within').textContent = `${(100 * data.test.within_one).toFixed(1).replace('.', ',')} %`;
  el('submit').disabled = false; el('submit').textContent = 'Estimer kvalitet ↗';
}).catch(error => { el('form-error').textContent = `${error.message} Prøv å laste sida på nytt.`; });
el('wine-form').addEventListener('input', clearResult);
el('reset').addEventListener('click', () => {
  if (!artifact) return;
  artifact.features.forEach((f, i) => { el(`f${i}`).value = f.default; });
  clearResult(); el('form-error').textContent = '';
});
el('wine-form').addEventListener('submit', event => {
  event.preventDefault();
  if (!artifact || !event.target.reportValidity()) return;
  const row = artifact.features.map((_, i) => el(`f${i}`).valueAsNumber);
  if (row.some(value => !Number.isFinite(value) || value < 0)) {
    el('form-error').textContent = 'Alle målingane må vere endelege, positive tal eller null.'; return;
  }
  el('form-error').textContent = '';
  const out = row.some((x, i) => x < artifact.features[i].min || x > artifact.features[i].max);
  const score = Vinlab.predict(artifact.model, row);
  el('score').textContent = score.toFixed(2).replace('.', ',');
  el('result-label').textContent = 'Estimert kvalitet';
  el('result-description').textContent = 'Modellen sitt estimat av smakspanelet si vurdering, på ein skala frå 0 til 10.';
  el('marker').style.left = `${Math.max(0, Math.min(100, score * 10))}%`;
  el('marker').style.opacity = '1';
  el('warning').textContent = out ? 'Minst ei måling ligg utanfor treningsintervallet. Tolkinga er særleg usikker.' : '';
});
