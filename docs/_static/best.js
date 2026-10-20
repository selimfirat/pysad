// Makes the best value of each metric column bold in the tables with the class "best-columns".
// The lowest value is best in a column of ranks or times, and the highest value in the others.
document.addEventListener("DOMContentLoaded", () => {
  const metric = /AUROC|AUPR|rank|Time|Standard|Low false alerts|Low misses|Solved|Precision|Recall|F1/;
  for (const table of document.querySelectorAll("table.best-columns")) {
    const headers = [...table.querySelectorAll("thead th")].map((th) => th.textContent);
    const rows = [...table.querySelectorAll("tbody tr")].map((tr) => [...tr.children]);
    headers.forEach((header, i) => {
      if (!metric.test(header)) return;
      const values = rows.map((cells) => parseFloat(cells[i].textContent));
      const best = /rank|Time/.test(header) ? Math.min(...values.filter(isFinite)) : Math.max(...values.filter(isFinite));
      rows.forEach((cells, row) => {
        if (values[row] === best) cells[i].style.fontWeight = "bold";
      });
    });
  }
});
