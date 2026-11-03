import React from "react";
import { BarChart3 } from "lucide-react";
export function DataChart({ charts, canShow }) {
  if (!canShow) {
    return (
      <section className="chart-panel">
        <div className="section-heading">
          <div className="panel-heading">
            <BarChart3 size={20} />
            <div>
              <h2>Data Visualization</h2>
              <p>Generated after analysis and report completion.</p>
            </div>
          </div>
        </div>
        <p className="detail">Visualization will appear after the agents complete the full analysis.</p>
      </section>
    );
  }

  if (!charts.length) {
    return (
      <section className="chart-panel">
        <div className="section-heading">
          <div className="panel-heading">
            <BarChart3 size={20} />
            <div>
              <h2>Data Visualization</h2>
              <p>Generated after analysis and report completion.</p>
            </div>
          </div>
        </div>
        <p className="detail">The analysis finished, but no chart images were generated for this dataset.</p>
      </section>
    );
  }

  return (
    <section className="chart-panel">
      <div className="section-heading">
        <div className="panel-heading">
          <BarChart3 size={20} />
          <div>
            <h2>Data Visualization</h2>
            <p>Charts generated from the analyzed dataset.</p>
          </div>
        </div>
      </div>

      <div className="generated-chart-grid">
        {charts.map((chart) => (
          <article className="generated-chart-card" key={chart.image_url}>
            <h3>{chart.title}</h3>
            <img className="generated-chart-image" src={chart.image_url} alt={chart.title} />
            <p className="generated-chart-caption">{chart.caption}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
