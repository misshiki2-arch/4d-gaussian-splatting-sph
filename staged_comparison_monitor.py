"""B-only process observation delta, loaded into support.monitor's scoped globals.

Reuse the immutable monitor's identity, resource sample/predicate, exception and
clock helpers. The driver retains its stop/reap/log publication contract.
No trainer, native extension, GPU probe, or historical helper mutation.
"""


class ObservationError(RuntimeError):
    def __init__(self, context):
        self.context = context
        super().__init__(context['detail'])


def failure_context(error, operation, role, pid):
    if isinstance(error, ObservationError):
        return dict(error.context)
    frames = traceback.extract_tb(error.__traceback__)[-3:]
    return dict(operation=operation, role=role, pid=pid,
        error_type=type(error).__name__, errno=getattr(error, 'errno', None),
        detail=str(error)[:160],
        location=[dict(file=Path(f.filename).name, line=f.lineno, function=f.name)
                  for f in frames])


def observe(operation, role, pid, function, *args):
    try:
        return function(*args)
    except Exception as error:
        raise ObservationError(failure_context(error, operation, role, pid)) from error


def disappeared(error):
    return isinstance(error, ObservationError) and error.context['errno'] in (errno.ENOENT, errno.ESRCH)


def member_identity(pid, role, expected=None):
    """One bounded re-observation, never permission/I/O/schema fallback."""
    try:
        row = observe('proc_identity', role, pid, proc_identity, pid)
    except ObservationError as first:
        if not disappeared(first):
            raise
        try:
            row = observe('proc_identity_recheck', role, pid, proc_identity, pid)
        except ObservationError as second:
            if disappeared(second):
                return None
            second.context['preceding'] = first.context
            raise
    if expected is not None and not same_process(expected, row):
        observe('identity_match', role, pid,
                lambda: (_ for _ in ()).throw(ValueError('process_identity')))
    return row


def group_members(owner, known):
    """Prove the owner/known members separately from unassigned enumeration."""
    if observe('boot_id', 'leader', owner['pid'], boot_id) != owner['boot_id']:
        observe('boot_match', 'leader', owner['pid'],
                lambda: (_ for _ in ()).throw(ValueError('boot_identity')))
    leader = member_identity(owner['pid'], 'leader', owner)
    found = {}
    if leader is not None:
        found[leader['pid']] = leader
    # Do not let a disappeared /proc entry silently erase a known member.
    # Reused IDs or members escaping the owned group fail even with a live leader.
    for pid, previous in list(known.items()):
        if pid == owner['pid']:
            continue
        row = member_identity(pid, 'known_member', previous)
        if row is not None:
            found[pid] = row
    paths = observe('proc_enumerate', 'unassigned', None, lambda: list(Path('/proc').iterdir()))
    for path in paths:
        if not path.name.isdigit() or int(path.name) in known:
            continue
        pid = int(path.name)
        try:
            raw = observe('candidate_stat', 'unassigned', pid, (path/'stat').read_text)
        except ObservationError as error:
            if disappeared(error):
                continue
            raise
        def parse():
            f = raw[raw.rindex(')')+2:].split()
            return dict(boot_id=owner['boot_id'], pid=pid, state=f[0],
                        pgid=int(f[2]), sid=int(f[3]), start_ticks=int(f[19]))
        stat = observe('candidate_stat_parse', 'unassigned', pid, parse)
        if stat['pgid'] != owner['pgid'] or stat['sid'] != owner['sid']:
            continue
        row = member_identity(pid, 'candidate_member')
        if row is None:
            continue
        def check():
            if (any(row[k] != stat[k] for k in ('pid','pgid','sid','start_ticks','boot_id'))
                    or row['uid'] != owner['uid'] or row['start_ticks'] < owner['start_ticks']):
                raise ValueError('member_identity')
            if leader is None:
                raise ValueError('unproven_remaining_member')
        observe('member_match', 'candidate_member', pid, check)
        known[pid] = row
        found[pid] = row
    return list(found.values())


def reconcile_terminal(p, owner, error):
    """Missing target data is not a resource sample: require actual child exit."""
    code = p.poll()
    if code is None:
        current = observe('terminal_identity', 'target', p.pid, proc_identity, p.pid)
        if not same_process(owner, current) or current['state'] != 'Z':
            raise error
        code = p.wait()  # Verified same-child Z, not a training wait.
    if code is None:
        raise error
    return code


def monitored_child(command, *, cwd, env, out, mode, filesystem,
                    interval=2.0, grace=5.0, kill_wait=5.0, sample_fn=None):
    """Same child lifecycle; role-aware observations and bounded failure context."""
    out = Path(out); sample_fn = resource_sample if sample_fn is None else sample_fn
    start = time.monotonic(); secondary = []; primary = None; owner = None
    known = {}; last = None; samples = 0; actions = []; remaining = None
    stop_confirmed = False; p = None
    stdout_path = out/(mode+'.stdout.log'); stderr_path = out/(mode+'.stderr.log')
    monitor_path = out/(mode+'-monitor.jsonl')
    with monitor_path.open('x') as journal, stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
        def context(error, operation):
            return failure_context(error, operation, 'target', p.pid if p else None)

        def emit(event, **fields):
            row = dict(event=event, at=now(), seconds=time.monotonic()-start)
            row.update(fields)
            try:
                journal.write(json.dumps(row, allow_nan=False)+'\n')
                journal.flush(); os.fsync(journal.fileno())
                return True
            except Exception as error:
                secondary.append(dict(stage='journal_'+event, **context(error, 'journal_'+event)))
                print('MONITOR_RECORD_UNAVAILABLE '+event+' '+type(error).__name__,file=sys.stderr,flush=True)
                return False

        p = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            owner = observe('spawn_identity', 'leader', p.pid, proc_identity, p.pid)
            if owner['pgid'] != p.pid or owner['sid'] != p.pid or owner['uid'] != os.getuid():
                raise ValueError('spawn_session_identity')
            known[p.pid] = owner
            if not emit('started', identity=owner, argv=command, cwd=str(cwd), limits=LIMITS, mode=mode):
                raise OSError('start_record')
            while True:
                # Discover before reap so new members require the live owner.
                members = group_members(owner, known)
                code = p.poll()
                if code is not None:
                    members = group_members(owner, known)
                    live = [r for r in members if r['state'] != 'Z']
                    if live:
                        primary = dict(kind='remaining_children', exit_code=code)
                    elif code != 0:
                        primary = dict(kind='child_exit', exit_code=code)
                    else:
                        stop_confirmed = True; remaining = members
                    break
                try:
                    values = sample_fn(owner, filesystem)
                except TerminalObservation as terminal:
                    if not same_process(owner, terminal.observed) or terminal.observed['state'] != 'Z':
                        raise ValueError('terminal_identity')
                    code = p.poll()
                    if code is None:
                        code = p.wait()
                    if code is None:
                        raise ProcessLookupError('terminal_unreaped')
                    if not emit('terminal_reconciled', identity=terminal.observed, exit_code=code):
                        raise OSError('terminal_record')
                    continue
                except Exception as error:
                    problem = ObservationError(context(error, 'resource_sample'))
                    if not disappeared(problem):
                        raise problem from error
                    try:
                        code = reconcile_terminal(p, owner, problem)
                    except Exception as recheck:
                        if recheck is not problem:
                            problem.context['reconciliation_error'] = context(recheck, 'terminal_recheck')
                        raise problem from error
                    if not emit('terminal_reconciled', identity=owner, exit_code=code,
                                observation=problem.context):
                        raise OSError('terminal_record')
                    continue
                bad = observe('resource_predicate', 'target', p.pid, violations, values, mode)
                last = values; samples += 1
                if bad:
                    primary = dict(kind='resource_budget', violations=bad, values=values, units='bytes', limits=LIMITS)
                if not emit('sample', values=values, units='bytes', members=[{k:r[k] for k in KEYS} for r in members]):
                    if primary is None: primary = dict(kind='monitor_record_failure')
                    break
                if bad: break
                try: p.wait(timeout=interval)
                except subprocess.TimeoutExpired: pass
        except Exception as error:
            primary = dict(kind='monitor_unavailable', **context(error, 'monitor'))

        if primary is not None:
            # Preserve child's observed nonzero exit alongside any monitoring error.
            primary['child_exit_code'] = p.poll()
            request = dict(primary=primary, at=now(), seconds=time.monotonic()-start, identity=owner, last_sample=last)
            try: write_json(out/(mode+'-stop-request.json'), request)
            except Exception as error:
                secondary.append(dict(stage='stop_request', **context(error, 'stop_request_write')))
                print('STOP_RECORD_UNAVAILABLE '+json.dumps(request)+' '+type(error).__name__,file=sys.stderr,flush=True)
            emit('stop_request', **request)
            if owner is not None:
                try:
                    for sig, wait_seconds in ((signal.SIGTERM, grace), (signal.SIGKILL, kill_wait)):
                        members = group_members(owner, known)
                        live = [r for r in members if r['state'] != 'Z']
                        if not live:
                            remaining = members; break
                        action = dict(signal=sig.name, at=now(), sent=False)
                        emit('signal_request', signal=sig.name)
                        try:
                            os.killpg(owner['pgid'], sig); action['sent'] = True
                        except ProcessLookupError:
                            action['already_absent'] = True
                        actions.append(action); emit('signal_result', **action)
                        deadline = time.monotonic()+wait_seconds
                        while True:
                            remaining = group_members(owner, known)
                            p.poll()
                            if not any(r['state'] != 'Z' for r in remaining) or time.monotonic() >= deadline:
                                break
                            time.sleep(min(0.1, max(0.001, deadline-time.monotonic())))
                        if not any(r['state'] != 'Z' for r in remaining): break
                    remaining = group_members(owner, known)
                    stop_confirmed = not any(r['state'] != 'Z' for r in remaining)
                except Exception as error:
                    secondary.append(dict(stage='stop_identity_or_signal', **context(error, 'stop_identity_or_signal')))
                    emit('stop_unconfirmed', secondary=secondary[-1])
            try: p.wait(timeout=kill_wait)
            except subprocess.TimeoutExpired as error:
                stop_confirmed = False
                secondary.append(dict(stage='wait', **context(error, 'child_wait')))
            emit('stop_result', confirmed=stop_confirmed, exit_code=p.returncode, remaining=remaining)
        else:
            p.wait()
        record = dict(argv=command, identity=owner, exit_code=p.returncode, timed_out=False,
            seconds=time.monotonic()-start, primary=primary, signals=actions, stop_confirmed=stop_confirmed,
            remaining=remaining, secondary_errors=secondary, samples=samples, last_sample=last)
        if not emit('finished', exit_code=p.returncode, primary=primary, stop_confirmed=stop_confirmed):
            if primary is None: record['primary'] = dict(kind='monitor_record_failure')
    try:
        record['logs'] = {s:identity(out/(mode+'.'+s+'.log')) for s in ('stdout','stderr')}
        record['monitor'] = identity(monitor_path)
        write_json(out/(mode+'-exit.json'), record)
    except Exception as error:
        if record['primary'] is None: record['primary'] = dict(kind='exit_record_failure')
        record['secondary_errors'].append(dict(stage='exit_record', **context(error, 'exit_record_write')))
        print('EXIT_RECORD_UNAVAILABLE '+type(error).__name__, file=sys.stderr, flush=True)
    if record['primary'] is not None or not stop_confirmed:
        if record['primary'] is None: record['primary'] = dict(kind='stop_unconfirmed')
        raise ChildFailure(record)
    return record
