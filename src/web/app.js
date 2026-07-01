console.log('[vdb.rust] page loaded');

document.getElementById('load-example').addEventListener('click', () => {
    const dim = 768;
    const vec = Array.from({ length: dim }, () => Math.random());
    document.getElementById('query').value = JSON.stringify(vec);
    console.log('[vdb.rust] action: load-example', { dim });
});

document.getElementById('search').addEventListener('click', async () => {
    const query = document.getElementById('query').value;
    console.log('[vdb.rust] action: search', { query: query.slice(0, 60) + '...' });
    const result = { status: 'placeholder', message: 'Search not yet implemented' };
    document.getElementById('search-result').textContent = JSON.stringify(result, null, 2);
    console.log('[vdb.rust] response:', result);
});

document.getElementById('run-benchmark').addEventListener('click', async () => {
    console.log('[vdb.rust] action: run-benchmark');
    const result = { status: 'placeholder', message: 'Benchmark not yet implemented' };
    document.getElementById('benchmark-result').textContent = JSON.stringify(result, null, 2);
    console.log('[vdb.rust] response:', result);
});
