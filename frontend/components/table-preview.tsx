export default function TablePreview({ rows }: { rows: string[][] }) {
  if (!rows.length) return null;
  const [header, ...body] = rows;

  return (
    <div className="overflow-x-auto rounded-md border border-ink-600">
      <table className="w-full min-w-max border-collapse text-left text-[13px]">
        <thead>
          <tr className="bg-ink-700">
            {header.map((cell, i) => (
              <th
                key={i}
                className="whitespace-nowrap border-b border-ink-600 px-3 py-1.5 font-mono text-[11px] font-medium uppercase tracking-wide text-brass-soft"
              >
                {cell || "—"}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {body.map((row, ri) => (
            <tr key={ri} className={ri % 2 === 0 ? "bg-ink-800" : "bg-ink-800/60"}>
              {row.map((cell, ci) => (
                <td key={ci} className="whitespace-nowrap border-b border-ink-700 px-3 py-1.5 text-paper/90">
                  {cell || "—"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
