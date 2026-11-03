import { CheckCircle2, CircleDashed, Clock3, Cpu } from "lucide-react";
import React from "react";

const agents = [
  "Data Preprocessing Agent",
  "Model Selection and Training Agent",
  "Model Evaluation Agent",
  "Visualization Agent",
  "Insight Agent",
  "Report Agent",
];

export function AgentTimeline({ results, isRunning }) {
  return (
    <section className="timeline-section">
      <div className="section-heading">
        <div className="panel-heading">
          <Cpu size={20} />
          <div>
            <h2>Agent Progress</h2>
            <p>Six deployed agents execute the analytics pipeline in sequence.</p>
          </div>
        </div>
        <span>{results.length} / {agents.length}</span>
      </div>

      <div className="timeline">
        {agents.map((agent, index) => {
          const result = results.find((item) => item.title === agent);
          const active = isRunning && !result && results.length === index;
          return (
            <article className={`agent-card ${result ? "complete" : active ? "active" : ""}`} key={agent}>
              <div className="agent-card-header">
                {result ? <CheckCircle2 size={20} /> : active ? <Clock3 size={20} /> : <CircleDashed size={20} />}
                <div>
                  <h3>{agent}</h3>
                  <p>{result?.status || (active ? "running" : "waiting")}</p>
                </div>
              </div>
              <p className="agent-output">{result?.output || "Waiting for upstream context."}</p>
            </article>
          );
        })}
      </div>
    </section>
  );
}
